from typing import List, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops
from torchvision.ops.roi_pool import roi_pool
from torchvision.models.resnet import conv1x1, conv3x3


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv3x3(width, width, stride, groups, dilation)
        self.bn3 = norm_layer(width)
        self.conv4 = conv1x1(width, planes * self.expansion)
        self.bn4 = norm_layer(planes * self.expansion)        
        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gelu(out)        

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
# class SwinTransformerBlock(nn.Module):


# class SparseAttnBlock(nn.Module):


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, intermdeiate_channels, num_classes):
        super(TwoMLPHead, self).__init__()

        self.fc1 = nn.Linear(in_channels, intermdeiate_channels)
        self.fc2 = nn.Linear(intermdeiate_channels, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class RoIPool(nn.Module):    
    """
    See :func:`roi_pool`.
    """

    def __init__(self, output_size: BroadcastingList2[int], spatial_scale: float):
        super(RoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr    