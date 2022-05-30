import os
import timm
import logging
import argparse
import pandas as pd
from typing import Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Bottleneck, TwoMLPHead, SparseAttnBlock
from torchvision.ops.roi_pool import RoIPool
from torchvision.ops.roi_align import RoIAlign


class PapsClassificationModel(nn.Module):
    def __init__(self, arch, pretrained, num_classes) :
        super(PapsClassificationModel, self).__init__()
        # get just top feature map only
        self.arch = arch
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.midplanes = 512
        
        self.model = timm.create_model(self.arch, pretrained=self.pretrained, num_classes=0, global_pool='')
        backbone_state_dict = self.model.state_dict()
        in_features = timm.create_model(self.arch, pretrained=False, num_classes=self.num_classes).get_classifier().in_features 
        self.conv = nn.Conv2d(in_features, self.midplanes, kernel_size=1, stride=1, bias=False)
        
        # use conv (bottleneck from resnet)
        # self.interlayer = Bottleneck(self.midplanes, self.midplanes)
        
        # use local + global attn block
        self.interlayer = SparseAttnBlock(self.midplanes, self.midplanes)
        
        # self.roi_pool = RoIPool((1,1), float(1/32)) # float(1/32), boxes is not normalized
        self.roi_pool = RoIAlign((1,1), float(1/32), sampling_ratio=2)

        self.pool_size = 5
        self.mlp = TwoMLPHead(self.midplanes, self.midplanes, self.num_classes*self.pool_size)   
        self.maxpool = nn.MaxPool1d(self.pool_size, stride=self.pool_size)
        
        # linear layer is initialized by default
        for m in self.modules():
            self.init_layer(m)  
            
        # load pretrained model
        if self.pretrained :
            print('load pretrained model')
            self.model.load_state_dict(backbone_state_dict)
            pass
                
    def init_layer(self, m) :
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)          
                
        
    def forward(self, x, boxes):
        batch_size = x.shape[0]
        x = self.model(x)
        x = self.conv(x)
        x = self.interlayer(x)   
        
        boxes = [ b.squeeze(dim=0) for b in torch.split(boxes, 1, dim=0)]
        
        x = self.roi_pool(x, boxes)

        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        
        # x = self.maxpool(x.view(batch_size, self.num_classes, -1)).squeeze(dim=-1)
        x = self.maxpool(x)
        
        return x

        '''
        roi_list = []
        
        for i in range(batch_size) :
            roi = self.roi_pool(x[i].unsqueeze(dim=0), [boxes[i]])
            roi_list.append(roi)
            
        roi_all = torch.cat(roi_list, dim=0)
        outputs = self.mlp(roi_all)  
        return outputs   
        '''