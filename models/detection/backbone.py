import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from typing import Dict, List, Optional, Callable
import timm

from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock, LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

# from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.detection.backbone_utils import BackboneWithFPN
# from torchvision.models.detection.retinanet import RetinaNet

# use timm instead of torchvision, because timm has more models
class PapsBackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels_list: List[int] = [],
        out_channels: int = 256,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ) -> None:
        super(PapsBackboneWithFPN, self).__init__()
        
        if not backbone  :
            raise ValueError(
                "backbone should be set, for example resnet18"
            )            

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        
        self.out_channels = out_channels
        self.extra_block = LastLevelP6P7(out_channels,out_channels)

        self.body = timm.create_model(backbone, features_only=True)
        self.in_channels_list = self.body.feature_info.channels()[2:]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=self.extra_block,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        x = self.body(x)
        out['0'] = x[-3]
        out['1'] = x[-2]
        out['2'] = x[-1]        
        out = self.fpn(out)
        # print(out.keys())
        return out