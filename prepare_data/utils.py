import os
import pandas as pd
import numpy as np
import sys
import cv2
import re
import json
import random

from xml.etree.ElementTree import parse
from sklearn.model_selection import GroupShuffleSplit, train_test_split

CLASS_MAPPER = {
    # [DOAI]
    # ASC-US
    "ASC-US": "ASC-US",
    "ASCUS-SIL": "ASC-US",
    "ASC-US with HPV infection": "ASC-US",
    # ASC-H
    "ASC-H": "ASC-H",
    "ASC-H with HPV infection": "ASC-H",
    # LSIL
    "LSIL": "LSIL",
    "LSIL with HPV infection": "LSIL",
    # HSIL
    "HSIL": "HSIL",
    "H": "HSIL",
    "HSIL with HPV infection": "HSIL",
    # Carcinoma
    "Carcinoma": "Carcinoma",
    
    # [SCL]
    # ASC-US
    "AS": "ASC-US",
    # ASC-H
    "AH": "ASC-H",
    # LSIL
    "LS": "LSIL",
    # HSIL
    "HS": "HSIL",
    "HN": "HSIL",
    # Carcinoma
    "SM": "Carcinoma",
    "SC": "Carcinoma",
    "C": "Candida",
    "Negative": 'Negative',
    "판독불가" : 'Negative',
    "Candida" : 'Candida',
    "Benign atypia" : 'Benign',
    "N - Endometrial cell" : 'Endometrial',
    "N - Endocervical Cell" : 'Endocervical',
    "Endocervical" : "Endocervical",
    "Endometrial" : "Endometrial",
    'Benign' : 'Benign',
    'Candida' : 'Candida',
}


def get_upper_dir(num) :
    upper_dir = ''
    if num > 0 and num <= 40 :
        upper_dir = '2021.01.06/'
    elif num <= 86 :
        upper_dir = '2021.01.07/'
    elif num <= 143 :
        upper_dir = '2021.01.08/'
    elif num <= 205 :
        upper_dir = '2021.01.11/'
    elif num <= 312 :
        upper_dir = '2021.01.12/'   
    elif num <= 360 :
        upper_dir = '2021.01.13/'
    elif num <= 428 :
        upper_dir = '2021.01.14/' 
    elif num <= 523 :
        upper_dir = '2021.01.15/'
    elif num <= 608 :
        upper_dir = '2021.05.11/'    
    elif num <= 681 :
        upper_dir = '2021.05.12/'
    elif num <= 748 :
        upper_dir = '2021.05.17/' 
    elif num <= 813 :
        upper_dir = '2021.05.18/'
    elif num <= 892 :
        upper_dir = '2021.05.20/'          
    elif num <= 952 :
        upper_dir = '2021.05.21/'
    elif num <= 1021 :
        upper_dir = '2021.05.24/' 
    elif num <= 1098 :
        upper_dir = '2021.05.25/'
    elif num <= 1173 :
        upper_dir = '2021.05.26/'  
    elif num <= 1245 :
        upper_dir = '2021.05.27/' 
    elif num <= 1351 :
        upper_dir = '2021.05.28/'
    elif num <= 1428 :
        upper_dir = '2021.05.31/'   
    elif num <= 1485 :
        upper_dir = '2021.06.01/'    
    elif num <= 1581 :
        upper_dir = '2021.06.02/'   
    elif num <= 1640 :
        upper_dir = '2021.06.03/'  
    elif num <= 1699 :
        upper_dir = '2021.06.04/'   
    elif num <= 1794 :
        upper_dir = '2021.06.07/'   
    elif num <= 1857 :
        upper_dir = '2021.06.10/'   
    elif num <= 1947 :
        upper_dir = '2021.06.11/'   
    elif num <= 2046 :
        upper_dir = '2021.06.14/'   
    elif num <= 2119 :
        upper_dir = '2021.06.15/'   
    elif num <= 2170 :
        upper_dir = '2021.06.16/'    
    elif num <= 2241 :
        upper_dir = '2021.06.17/'   
    elif num <= 2316 :
        upper_dir = '2021.06.18/'
        
    return 'patch_images/' + upper_dir

def anno_to_df(df, src_path):
    file_names = []
    task_names = []
    labels = []
    xmins = []
    ymins = []
    ws = []
    hs = []
    occludeds = []
    dess = []
    cell_types = []
    
    xml_list = [d +'/annotations.xml' for d in os.listdir(src_path) if not d.endswith('.zip')]
    for xml_path in xml_list :
        if not 'ipynb_checkpoints' in xml_path :
            parser = parse(src_path + xml_path)
            root = parser.getroot()
            task_name = root.find("meta").find("task").find("name").text
            print(task_name)
            images = root.findall("image")

            for image in images:
                file_name = image.get("name")
                if 'scl' in file_name :
                    task_name = file_name.split("scl/")[-1].split('-')[0]
                    task_number = int(re.sub('LBC', '', task_name))
                    uppder_dir = get_upper_dir(task_number)
                    file_name = uppder_dir + re.sub('scl/', '', file_name)     
                elif 'negative' in file_name :
                    task_name = file_name.split("negative/")[-1].split('-')[0]
                    task_number = int(re.sub('LBC', '', task_name))
                    uppder_dir = get_upper_dir(task_number)
                    file_name = uppder_dir + re.sub('negative/', '', file_name)   
                elif 'carcinoma' in file_name :
                    task_name = file_name.split("carcinoma/")[-1].split('-')[0]
                    task_number = int(re.sub('LBC', '', task_name))
                    uppder_dir = get_upper_dir(task_number)
                    file_name = 'patch_images1/' + re.sub('carcinoma/', '2022.04.08/', file_name)                         

                elif len(file_name.split('/')) ==  3 :
                    file_name = 'patch_images/' + file_name

                bboxes = image.findall("box")
                if len(bboxes) > 0:
                    for bbox in bboxes:
                        label = bbox.get('label')
                        xmin = float(bbox.get('xtl'))
                        ymin = float(bbox.get('ytl'))
                        xmax = float(bbox.get('xbr'))
                        ymax = float(bbox.get('ybr'))
                        occluded = bbox.get('occluded')
                        if not bbox.find('attribute') is None :
                            if bbox.find('attribute').get('name') == 'description' :
                                des = bbox.find('attribute').text
                                cell_type = ''
                            else :
                                cell_type = bbox.find('attribute').text
                                des = ''
                        else :
                            cell_type = ''
                            des = ''                        

                        file_names.append(file_name)
                        task_names.append(task_name.split('-2')[0])
                        labels.append(label)
                        xmins.append(xmin)
                        ymins.append(ymin)
                        ws.append(xmax-xmin)
                        hs.append(ymax-ymin)
                        occludeds.append(occluded)
                        dess.append(des)
                        cell_types.append(cell_type)
                        
    df.file_name = file_names
    df.task = task_names
    df.label = labels
    df.xmin = xmins
    df.ymin = ymins
    df.w = ws
    df.h = hs
    df.occluded = occludeds
    df.des = dess
    df.cell_type = cell_types
    
    return df

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
# df to json        
def convert(categories, df, json_file="test.json"):
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    bnd_id = 1
    image_id=0
    
    for img_id in df.ID.unique():
        image_id +=1
        height, width=2048,2048
        temp_df = df[df['ID']== img_id]
        temp_df.reset_index(drop=True, inplace=True)  
        image = {'file_name': temp_df.file_name.unique()[0], 
                 'height': height, 
                 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)

        for row in range(len(temp_df)):           
            # category = temp_df.iloc[row,9] #label id columns
            # df.loc[1][ ['task','label']]
            category = temp_df.loc[row]['label_det_one'] #label_det_one columns for one class detection
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            # box = temp_df.iloc[row,4:8]
            box = temp_df.loc[row][['xmin','ymin','w','h']]
            if len(box) > 0:
                xmin,ymin,w,h = list(box)
                xmax = xmin + w
                ymax = ymin + h

                assert(xmax > xmin)
                assert(ymax > ymin)
                
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': w*h, 'iscrowd':0, 'image_id':
                       image_id, 'bbox': [xmin, ymin, w, h],
                       'category_id': category_id, 'id': bnd_id, 'ignore': 0}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_file, 'w',encoding='utf-8')
    json_str = json.dumps(json_dict,cls=NpEncoder)
    json_fp.write(json_str)
    json_fp.close()   
    
    
def drop_wrong(df, columns='label_cls') :
    '''    
    # remove wrong annotations 
    # check this notebook files, clf_from_roi/prepare_data/notebooks/check_wrong.ipynb

    # annotaion guide line document
    # https://doaiacropolis.atlassian.net/wiki/spaces/DOP/pages/1885700101/SCL
    
    # return Negative, ASC-H, ASC-US, HSLL, LSIL, carcinorma, total six classes
    '''    
    
    df[columns] = df.label.apply(lambda x : CLASS_MAPPER[str(x)])
    
    # check absolute size of w and h, delete abnormally small size 
    df = df[(df['w'] > 10) & (df['h'] > 10)]
    
    # check aspect ratio and delete abnormal, most box is inside this range except candida,
    df = df[(df['ratio'] < 2.5) & (df['ratio'] > 0.4)]
    
    # remove grouped annotated cell for negative
    group_Neg_df = df[(df[columns] == 'Negative') & (df['area'] > 250)]
    df = pd.concat([df, group_Neg_df]).drop_duplicates(keep=False)
    
    # remove Candida or C
    df = df[(df[columns] != 'Candida') & (df[columns] != 'C')]
    
    # remove Benign atypia, Benign,  this is not included in 6 class, 
    df = df[(df[columns] != 'Benign atypia') & (df[columns] != 'Benign')]    
    
    # remove N - Endocervical Cell, Endometrial
    df = df[(df[columns] != 'N - Endocervical Cell')] 
    df = df[(df[columns] != 'Endocervical') & (df[columns] != 'Endometrial')]    
    
    # remove ASCUS-SIL cell
    df = df[(df[columns] != 'ASCUS-SIL')]   
    
    # remove herpes cell
    df = df[(df[columns] != 'H')]         
    
    # remove abnormally big size samples
    df = df[(df['area'] < 1400)] 
    
    # remove abnormally small size samples
    df = df[(df['area'] > 20)]     
    
    # # remove Carcinoma temporalliy becuase there is few samples
    # df = df[(df[columns] != 'Carcinoma')]       
    
    return df


def split_by_group(df, ratio, seed ) :
    train_inds, test_inds = next(GroupShuffleSplit(test_size=ratio,
                                                    n_splits=2, 
                                                    random_state = seed).split(df, groups=df['task']))
    return train_inds, test_inds

def check_ratio(train_df, test_df, columns='label_det') :

    if len(train_df[columns].unique()) == len(test_df[columns].unique()) :
        ratio_dict = {}
        for label in train_df[columns].unique() :
            nom = len(test_df[test_df[columns] == label])
            denom = nom + len(train_df[train_df[columns] == label])
            ratio_dict[label] = nom/denom
            
        return ratio_dict
    return None


def split_by_group_check_ratio(df, test_ratio, seed, limit=0.05, columns='label_det' ) :
    TRAIN_TEST_SPLIT_PERC = 1 - test_ratio
    uniques = df["task"].unique()
    sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)
    
    for i in range(10000) :
        random.shuffle(uniques)
        train_ids, test_ids = uniques[:sep], uniques[sep:]
        train_df, test_df = df[df.task.isin(train_ids)], df[df.task.isin(test_ids)]
        ratio_dict = check_ratio(train_df, test_df, columns=columns) 
        
        flag = True
        if ratio_dict :
            for r in ratio_dict.keys() :
                if abs(ratio_dict[r] - test_ratio) < limit :
                    pass
                else :
                    flag = False
                    break
                    
        if flag == True and ratio_dict is not None:
            print(ratio_dict)
            return train_df.index, test_df.index
    

# split by group but consider real sample ratio, as much as possble
# because a task has multiple labels, task can be mixed
def split_by_group_ratio(tdf, test_ratio, seed ) :
    df = tdf.copy()
    TRAIN_TEST_SPLIT_PERC = 1- test_ratio
    uniques = tdf["task"].unique()
    sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)

    best_ratio = 0
    best_unique = []
    best_train_df = []
    best_test_df = []
    
    df = tdf.copy()
    
    for i in range(40) :
        random.shuffle(uniques)
        
        # df = df.sample(frac=1).reset_index(drop=True) #For shuffling your data
        train_ids, test_ids = uniques[:sep], uniques[sep:]
        train_df, test_df = df[df.task.isin(train_ids)], df[df.task.isin(test_ids)]
        ratio = len(test_df)/(len(test_df) + len(train_df))
        if ratio > best_ratio and ratio < test_ratio :
            best_ratio = ratio
            best_unique.append(uniques)
            best_train_df.append(train_df)
            best_test_df.append(test_df)

    print(len(best_test_df[-1])/(len(best_test_df[-1]) + len(best_train_df[-1])))
    
    return best_train_df[-1].index, best_test_df[-1].index

def paps_data_split (df, ratio=0.25, seed=0, method='both', columns='label_det') :
    if method == 'whole_slide' :
        train_inds, test_inds = split_by_group(df, ratio, seed)                                                       

    elif method == 'box' :
        # use original label but label_det
        targets = df['label']
        train_inds, test_inds = train_test_split(np.arrange(len(df)),
                                                 test_size=ratio,
                                                 stratify=targets,
                                                 random_state=seed)

    else :
        train_inds, test_inds = split_by_group_check_ratio(df, ratio, seed, limit=0.03, columns=columns)


    return train_inds, test_inds


def set_id(df) :
    # ID is used for coco evaluation instead of index
    id_dict = {}
    for i, f in enumerate(list(df.file_name.unique())) :
        id_dict[f] = i   
                          
    df['ID'] = df.file_name.apply(lambda x : id_dict[x])  
    
    return df



        


