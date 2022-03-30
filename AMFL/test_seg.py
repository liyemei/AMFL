# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:18:36 2020

@author: liyemei
"""

import argparse
import os
from math import log10
import numpy  as np
from util.test_overlap_dataset import test_DataFromH5File
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
from util.networks import define_G, define_D, print_network

# Testing settings
parser = argparse.ArgumentParser(description='torch-implementation')

parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=4, help='output image channels')
parser.add_argument('--ngf', type=int, default=16, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
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
    

device = torch.device("cuda:1" if train_on_gpu else "cpu")

print('===> Building model')
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False).to(device)



    
print('===> Loading datasets')
root_path = "../dataset/overlap_chromosome/Cleaned_LowRes_13434_overlapping_pairs.h5"
test_dataset = test_DataFromH5File(root_path,4 ,0.2)
test_data_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)

print('===> Begining Testing')


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask

def test():
   
    test_iterator = tqdm(test_data_loader,leave=True, total=len(test_data_loader),position=0,ncols=10)
    iterators=0
    netG.eval()
    for  inputs, labels in test_iterator:
        iterators=iterators+1
        inputs=inputs.to(device)
        labels=labels.to(device) 
        y_pred = netG(inputs)[0]
        output=y_pred.cpu().data

        output=output.numpy()
        gt=labels.cpu().data
        gt=gt.numpy()[0]
        mask=onehot2mask(output)
        mask_rgb=color.label2rgb(mask,bg_label=0)*255       
        gt=onehot2mask(gt)
        gt_rgb=color.label2rgb(gt,bg_label=0)*255
        if len(np.unique(gt))==4:
            cv2.imwrite("results/pred/{}.bmp".format(iterators), mask)
            cv2.imwrite("results/pred_vis/{}.bmp".format(iterators), mask_rgb.astype(np.uint8))
            cv2.imwrite("results/gt_vis/{}.bmp".format(iterators), gt_rgb.astype(np.uint8))
            cv2.imwrite("results/gt/{}.bmp".format(iterators), gt)
      
        status="===>Epoch[{}]".format(iterators)
        print(status)

if __name__ == '__main__' :  
        log_dir='checkpoint/model.pth'
        last_log='checkpoint/seq_last.pth'
      
        count=0
        if os.path.exists(log_dir):

            netG.load_state_dict(torch.load(log_dir,map_location={'cuda:0':'cuda:1'}))
            print('load_weight')
            netG.to(device)  
           
        test()
          

        
        

        

       
        
        
        
            
        
