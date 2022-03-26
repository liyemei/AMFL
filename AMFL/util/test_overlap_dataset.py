# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:33:38 2020

@author: AILab
"""
import numpy  as np
import glob
import os
import natsort
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
import warnings
#from scipy.misc import imread, imresize
warnings.filterwarnings("ignore")
import cv2
import h5py

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

class test_DataFromH5File(Dataset):
    def __init__(self, filepath,num_classes,percentage=0.2):
        f = h5py.File(filepath, 'r')
        keys=[key for key in f.keys()] 
        temp_data = f[keys[0]][:] 
        f.close()
        n=round(temp_data.shape[0]*(1-percentage))
        data=temp_data[n+1:]
        print ('===>[INFO]:{}% datasets was split to testing sets,a total of {}  images'.format(percentage*100, data.shape[0]))        
        
        
        self.img = data[:,:,:,0]
        self.gt = data[:,:,:,1]
        w,h=self.img[0].shape
        w_pad=(128-h)//2
        h_pad=(128-w)//2
        self.pading_img=[]
        self.pading_gt=[]
        for i in range( self.img.shape[0]):
          pad_img = np.pad(self.img[i], ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=255)
        
          pad_img=np.column_stack((pad_img, np.zeros(128)))/255.0  
          self.pading_img.append(pad_img)
		  
          _pad_gt = np.pad(self.gt[i], ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=0)
          _pad_gt=np.column_stack((_pad_gt, np.zeros(128)))
#          print(np.unique(_pad_gt),i)

          pad_gt= mask2onehot(  _pad_gt,num_classes)
          
          self.pading_gt.append(pad_gt)

#          
        print("===>[INFO]:processed pading img {}/{} done".format(i+1,self.img.shape[0]))
        self.pading_img=np.array( self.pading_img)
        self.pading_gt=np.array( self.pading_gt)
   
        
    def __getitem__(self, idx):
       
        img = np.expand_dims( self.pading_img, axis=1)
        gt =  self.pading_gt
        
        data = torch.from_numpy( img[idx]).float()
        
        label = torch.from_numpy(gt[idx]).float()

       
        return data,  label
    
    def __len__(self):
        assert self.img.shape[0] == self.gt.shape[0], "Wrong data length"
        return self.img.shape[0]


