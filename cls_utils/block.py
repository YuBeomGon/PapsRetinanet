from typing import List, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops
from torchvision.ops.roi_pool import roi_pool
from torchvision.models.resnet import conv1x1, conv3x3
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



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
        self.conv4 = conv1x1(width, planes)
        self.bn4 = norm_layer(planes)        
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = conv1x1(in_features, hidden_features)
        self.act = act_layer()
        self.conv2 = conv1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size=8, islocal=True):
    """
    Args:
        x: (B, C, H, W )
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    if islocal :
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
        windows = windows.view(-1, window_size * window_size, C)
    else :
        windows = x.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, window_size, window_size, C)
        windows = windows.view(-1, H//window_size * W//window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, islocal=True):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window sizewindow_size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, C, H, W)
    """
    
    if islocal :
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    else :
        B = int(windows.shape[0] / (window_size * window_size))
        x = windows.view(B, -1, window_size, window_size, H // window_size, W // window_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)        
        
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SparseAttnLayer(nn.Module):
    def __init__(self, inplanes, planes,  norm_layer=nn.BatchNorm2d, islocal=True):
        super().__init__()
        self.midplanes = 512 # inplanes//4 if inplanes == 2048 else inplanes
        self.window_size = 8
        self.num_heads = self.midplanes//32
        self.islocal = islocal
        self.attn = WindowAttention(self.midplanes,  (self.window_size, self.window_size), self.num_heads)
        # self.H = self.W = None
        self.mlp = Mlp(in_features=self.midplanes, hidden_features=self.midplanes, act_layer=nn.GELU, drop=0.)
        self.norm1 = norm_layer(self.midplanes)
        self.norm2 = norm_layer(self.midplanes)
        
        self.drop_path_rate = 0.1
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0. else nn.Identity()
        
    def forward(self, x) :
        shortcut = x
        x = self.norm1(x)
        B, C, H, W = x.shape
        x_windows = window_partition(x, self.window_size, islocal=self.islocal)
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, self.midplanes)
        
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.midplanes)
        x = window_reverse(attn_windows, self.window_size, H, W, islocal=self.islocal) # B, C, H, W
        x = shortcut + self.drop_path(x)
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
class SparseAttnBlock(nn.Module) :
    def __init__(self, inplanes=512, planes=512,  norm_layer=nn.BatchNorm2d) :
        super().__init__()
        self.localattn1 = SparseAttnLayer(inplanes, planes,  norm_layer, islocal=True)
        self.globalattn1 = SparseAttnLayer(inplanes, planes,  norm_layer, islocal=False)
        
        self.localattn2 = SparseAttnLayer(inplanes, planes,  norm_layer, islocal=True)
        self.globalattn2 = SparseAttnLayer(inplanes, planes,  norm_layer, islocal=False)  
        
    def forward(self, x) :
        x = self.localattn1(x)
        x = self.globalattn1(x)
        x = self.localattn2(x)
        x = self.globalattn2(x)
        
        return x

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