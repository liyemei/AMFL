# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:18:36 2020

@author: liyemei
"""

import argparse
import os
from math import log10
import numpy  as np
from overlap_dataset import *
from test_overlap_dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image
import glob
from torch import optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
import shutil
import random
from tqdm import tqdm
import cv2
import time
from skimage import data,segmentation,measure,morphology,color


# Testing settings
parser = argparse.ArgumentParser(description='torch-implementation')
parser.add_argument('--dataset', default='train', help='model')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()
print(opt)
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    random_seed = random.randint(1, 100)
    print('random_seed = ' + str(random_seed))
    print('CUDA is not available. Training on CPU')
else:
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.seed)
    print('CUDA is available. Training on GPU')
    

device = torch.device("cuda:0" if train_on_gpu else "cpu")


model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]

print('===> Loading datasets')
root_path = "dataset/overlap_chromosome/Cleaned_LowRes_13434_overlapping_pairs.h5"
test_dataset =test_DataFromH5File(root_path,4 ,'test',0.2)
test_data_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)

print('===> Begining Testing')


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask

def test(model_name,loss_name):
    test_net=model_name
#   map_location={'cuda:1': 'cuda:0'}
    model=torch.load('model/{}/{}_model.pth'.format(loss_name,model_name))
    if test_net=='R2U_Net' or test_net=='R2AttU_Net':
       print('===>  Testing {}/{}:  model.eval() off'.format(loss_name,test_net))
    else:
       model.eval()
       print('===>  Testing {}/{}:  model.eval() on'.format(loss_name,test_net))

    test_iterator = tqdm(test_data_loader,leave=True, total=len(test_data_loader),position=0,ncols=10)
    iterators=0
    for  inputs, labels in test_iterator:
        iterators=iterators+1
        inputs=inputs.to(device)
        labels=labels.to(device)
        y_pred = model(inputs)
        output=y_pred.cpu().data
        output=output.numpy()[0]
        gt=labels.cpu().data
        gt=gt.numpy()[0]
        mask=onehot2mask(output)
        mask_rgb=color.label2rgb(mask,bg_label=0)*255
        gt=onehot2mask(gt)
        gt_rgb=color.label2rgb(gt,bg_label=0)*255
        cv2.imwrite("results/{}/{}/pred/{}.bmp".format( loss_name,model_name,iterators), mask)
        cv2.imwrite("results/{}/{}/pred_vis/{}.bmp".format(loss_name,model_name, iterators), mask_rgb.astype(np.uint8))
        if model_name=="U_Net" and model_name=="mutil_dice":
           if len(np.unique(gt))==4:
              cv2.imwrite("results/gt/{}.bmp".format( iterators), gt)
              cv2.imwrite("results/gt_vis/{}.bmp".format( iterators), gt_rgb.astype(np.uint8))
        status="===>Epoch[{}]".format(iterators)
        test_iterator.set_description(status)
if __name__ == '__main__' :  

     model_names=['ENet','BiSeNetV1','BiSeNetV2','DeepLabV3+','FastFCN','U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net','NestedUNet','AFML']
    
     loss_names=['mutil_dice','CE','weight_dice','weight_CE', 'lovasz_softmax']
     for i in range(len(model_names)):
        print("\n===>  test model  {} /{} ".format(loss_names[1], model_names[i]))
        test(model_names[i],loss_names[1])
          

        
        

        

       
        
        
        
            
        
