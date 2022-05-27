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
from .block import Bottleneck, TwoMLPHead, RoIPool


class PapsClassificationModel(nn.Module):
    def __init__(self, arch, pretrained, num_classes) :
        super(PapsClassificationModel, self).__init__()
        # get just top feature map only
        self.arch = arch
        self.pretrained = pretrained
        self.num_classes = num_classes
        
        self.model = timm.create_model(self.arch, pretrained=self.pretrained, num_classes=0, global_pool='')
        in_features = timm.create_model(self.arch, pretrained=False, num_classes=self.num_classes).get_classifier().in_features 
        self.interlayer = Bottleneck(in_features, in_features//4)
        
        self.roi_pool = RoIPool((1,1), float(1/32)) # float(1/32), boxes is not normalized
        
        intermdeiate_channels = in_features//4
        self.mlp1 = TwoMLPHead(in_features, intermdeiate_channels, self.num_classes)   
        self.mlp2 = TwoMLPHead(in_features, intermdeiate_channels, self.num_classes)  
        
        self.maxpool = nn.MaxPool1d(2, stride=2)
        
        for m in self.interlayer.modules():
            self.init_layer(m)
                
        for m in self.mlp1.modules():
            self.init_layer(m)
            
        for m in self.mlp2.modules():
            self.init_layer(m)            
                
    def init_layer(self, m) :
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)          
                
        
    def forward(self, x, boxes):
        batch_size = x.shape[0]
        x = self.model(x)
        x = self.interlayer(x)   
        
        boxes = [ b.squeeze(dim=0) for b in torch.split(boxes, 1, dim=0)]
        
        roi = self.roi_pool(x, boxes)
        output1 = self.mlp1(roi)
        output2 = self.mlp2(roi)
        
        outputs = torch.stack([output1, output2], dim=2)
        outputs = self.maxpool(outputs).squeeze(dim=-1)
        
        return outputs

        '''
        roi_list = []
        
        for i in range(batch_size) :
            roi = self.roi_pool(x[i].unsqueeze(dim=0), [boxes[i]])
            roi_list.append(roi)
            
        roi_all = torch.cat(roi_list, dim=0)
        outputs = self.mlp(roi_all)  
        return outputs   
        '''