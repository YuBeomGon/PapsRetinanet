# this is for making csv file for classification, 6 class

import os
import argparse
import pandas as pd
import numpy as np
import sys
import cv2
import re

from utils import CLASS_MAPPER, convert, drop_wrong, paps_data_split, set_id

parser = argparse.ArgumentParser(description='get dataframe of annotations')
parser.add_argument('--saved_path', default='saved/', type=str, metavar='M',
                    help='path for saved results')
parser.add_argument('--ratio', default=0.25, type=float, metavar='M',
                    help='ratio')
parser.add_argument('--seed', default=1, type=int, metavar='M',
                    help='seed for re occurence')
parser.add_argument('--whole_slide', default=True, type=bool, metavar='M',
                    help='split by wsi or bbox')

if __name__ == "__main__":
    print('start to prepare csv file')
    args = parser.parse_args()
    
    for file in ['train.csv', 'test.csv'] :
    
        df = pd.read_csv(args.saved_path + file)

        df['label_cls'] = df.label.apply(lambda x : CLASS_MAPPER[str(x)])
        df['label_hpv'] = df.label.apply(lambda x : int(1) if 'HPV' in str(x) else int(0))

        df.reset_index(drop=True, inplace=True)    

        df.to_csv(args.saved_path + file, index=None)

    print('csv for train/test files was made')
