import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from typing import Dict, List, Optional, Callable
import timm

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock, LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import sigmoid_focal_loss
from utils.dataset import MAX_IMAGE_SIZE

from .backbone import PapsBackboneWithFPN

def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

class RetinaNetClsHead(RetinaNetClassificationHead):
    
    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
        super().__init__(in_channels, num_anchors, num_classes, prior_probability=prior_probability)
        
    def compute_loss(self, targets, head_outputs, matched_idxs, head_name='cls_logits', target_name='labels'):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs[head_name]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image[target_name][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)    

class RetinaNetMultiHead(nn.Module) :
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = RetinaNetClsHead(in_channels, num_anchors, num_classes[0])
        self.classification_head1 = RetinaNetClsHead(in_channels, num_anchors, num_classes[1])
        self.classification_head2 = RetinaNetClsHead(in_channels, num_anchors, num_classes[2])
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            "cls": self.classification_head.compute_loss(targets, head_outputs, matched_idxs, 
                                                                   head_name="cls_logits", target_name='labels'),
            "cls_five": self.classification_head1.compute_loss(targets, head_outputs, matched_idxs,
                                                                    head_name="cls_logits_five", target_name='labels_five'),
            "cls_hpv": self.classification_head2.compute_loss(targets, head_outputs, matched_idxs,
                                                                     head_name="cls_logits_hpv", target_name='labels_hpv'),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {"cls_logits": self.classification_head(x), 
                "cls_logits_five": self.classification_head1(x),
                "cls_logits_hpv": self.classification_head2(x),
                "bbox_regression": self.regression_head(x)}
    

class PapsRetinaNet(RetinaNet):
    def __init__(
        self,
        backbone,
        num_classes,
        # transform parameters
        min_size=MAX_IMAGE_SIZE,
        max_size=MAX_IMAGE_SIZE,
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
    ):   

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)    
        if head is None:
            head = RetinaNetMultiHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
            
        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            # Anchor parameters
            anchor_generator=anchor_generator,
            head=head,
            proposal_matcher=proposal_matcher,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
            topk_candidates=topk_candidates, 
        )
        
        
        


            
