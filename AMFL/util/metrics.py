#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:18:02 2022

@author: root
"""

import torch
import numpy as np
import os
import random

def get_JS(SR,GT):
    # JS : Jaccard similarity
    intersection = GT * SR
    JS = (float(intersection.sum())) / float((GT.sum() + SR.sum()-intersection.sum()))
   
    return JS

def mutil_IOU(SR,GT): 
    
    num,num_classes,_,_=GT.size()
    mean_IoU=0
    for n in range(num):
        tmp_total_IOU=0
        for i in range(num_classes):
            tmp_total_IOU+=get_JS(SR[n][i,:,:],GT[n][i,:,:])
        tmp_mean_IOU=tmp_total_IOU/num_classes 
        mean_IoU+=tmp_mean_IOU
    return mean_IoU/num