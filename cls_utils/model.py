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
from .block import Bottleneck, TwoMLPHead, RoIPool, SparseAttnBlock


class PapsClassificationModel(nn.Module):
    def __init__(self, arch, pretrained, num_classes) :
        super(PapsClassificationModel, self).__init__()
        # get just top feature map only
        self.arch = arch
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.midplanes = 512
        
        self.model = timm.create_model(self.arch, pretrained=self.pretrained, num_classes=0, global_pool='')
        in_features = timm.create_model(self.arch, pretrained=False, num_classes=self.num_classes).get_classifier().in_features 
        self.conv = nn.Conv2d(in_features, self.midplanes, kernel_size=1, stride=1, bias=False)
        
        # use conv (bottleneck from resnet)
        # self.interlayer = Bottleneck(self.midplanes, self.midplanes)
        
        # use local + global attn block
        self.interlayer = SparseAttnBlock(self.midplanes, self.midplanes)
        
        self.roi_pool = RoIPool((1,1), float(1/32)) # float(1/32), boxes is not normalized

        self.fc1 = nn.Linear(self.midplanes, self.num_classes)   
        self.fc2 = nn.Linear(self.midplanes, self.num_classes)  
        self.fc3 = nn.Linear(self.midplanes, self.num_classes)  
        
        self.maxpool = nn.MaxPool1d(2, stride=2)
        
        for m in self.modules():
            self.init_layer(m)  
            
        # load pretrained model
        if self.pretrained :
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
        output1 = F.relu(self.fc1(x))
        output2 = F.relu(self.fc2(x))
        output3 = F.relu(self.fc3(x))
        
        outputs = torch.stack([output1, output2, output3], dim=2)
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