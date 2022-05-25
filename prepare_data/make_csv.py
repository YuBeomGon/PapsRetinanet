# make the default csv file, this is used for train test split for both classification and detection

import os
import argparse
import pandas as pd
import numpy as np
import sys
import cv2
import re

from utils import anno_to_df

parser = argparse.ArgumentParser(description='get dataframe of annotations')
parser.add_argument('--src_path', metavar='DIR', default='./annotations/',
                    help='path to src_path (default: ./annotations/)')
parser.add_argument('--ratio', default=0.25, type=float, metavar='M',
                    help='ratio')
parser.add_argument('--seed', default=1, type=int, metavar='M',
                    help='seed for re occurence')
parser.add_argument('--wsi', default=True, type=bool, metavar='M',
                    help='split by wsi or bbox')
parser.add_argument('--saved_path', default='saved/', type=str, metavar='M',
                    help='path for saved results')


if __name__ == "__main__":
    
    args = parser.parse_args()
    df = pd.DataFrame(columns=['file_name', 'task', 'label', 'xmin', 'ymin', 
                               'w', 'h', 'occluded', 'des', 'cell_type'])    
    df = anno_to_df(df, args.src_path)
    
    # get bbox tab and diag length for sorting by size
    df.xmin = df.xmin.apply(lambda x : int(x))
    df.ymin = df.ymin.apply(lambda x : int(x))
    df.w = df.w.apply(lambda x : int(x))
    df.h = df.h.apply(lambda x : int(x))
    df['bbox'] = df.apply(lambda x : [x['xmin'], x['ymin'], x['w'], x['h']], axis=1)
    
    # take the sqrt root of area
    df['area'] = df.apply(lambda x : int(np.sqrt([x['w']][0] * [x['h']][0])) , axis=1)
    
    # take the aspect ratio, need to sample in dataloader considering this ratio and area
    df['ratio'] = df.apply(lambda x : float(np.sqrt([x['w']][0] / [x['h']][0])) , axis=1)
    
    # remove null
    df.label = df.label.apply(lambda x :x.strip())
    
    df.to_csv(args.saved_path + 'df.csv', index=None)
    
    print('default csv was made')
    
    
    
  
    
             
    