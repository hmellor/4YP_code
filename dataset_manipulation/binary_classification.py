# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:28:16 2019

@author: hejme
"""
from os.path import join as pjoin
import numpy as np
import scipy.misc as m
from tqdm import tqdm
import collections
import os
exists = os.path.isfile('/path/to/file')
files = collections.defaultdict(list)

for split in ["train", "val"]:
    path = pjoin("../datasets/VOCdevkit/VOC2011/ImageSets/Segmentation", split + ".txt")
    file_list = tuple(open(path, "r"))
    file_list = [id_.rstrip() for id_ in file_list]
    files[split] = file_list

for split in ["train", "val"]:
    binary_path = pjoin("../datasets/VOCdevkit/VOC2011/ImageSets/Segmentation", split + "_binary.txt")
    exists = os.path.isfile(binary_path)
    if exists:
        print("Segmentation file {}_binary.txt already exists".format(split))
        break
    else:
        print("Generating binary {} split".format(split))
        with open(binary_path, 'a') as file:
            for ii in tqdm(files[split]):
                fname = ii + ".png"
                target_path = pjoin("../datasets/VOCdevkit/VOC2011/SegmentationClass/pre_encoded", fname)
                target = m.imread(target_path)
                if np.unique(target).size <= 2:
                    file.write(ii + "\n")
