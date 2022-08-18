from __future__ import print_function
import argparse
import os
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util.networks import define_G, define_D, print_network
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from util.overlap_dataset import DataFromH5File
from util.test_overlap_dataset import test_DataFromH5File
from util.val_overlap_dataset import val_DataFromH5File
from util.lovasz_losses import  LovaszSoftmax
from util.MulticlassDice import MulticlassDiceLoss
from util.metrics import mutil_IOU

torch.manual_seed(123)
from torchsummary import summary
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='U-net-PyTorch-implementation')
parser.add_argument('--batchSize', type=int, default=48, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=4, help='output image channels')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=20, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--ngf', type=int, default=16, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
opt = parser.parse_args()   
print(opt) 

np.random.seed(opt.seed) 
            
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




print('===> Loading train datasets')
root_path = "../dataset/overlap_chromosome/Cleaned_LowRes_13434_overlapping_pairs.h5"
train_dataset =DataFromH5File(root_path,4,0.8,0.25)
train_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Loading val datasets')
val_dataset = val_DataFromH5File(root_path,4,0.8,0.25)
val_data_loader = DataLoader(dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Loading test datasets')
test_dataset = test_DataFromH5File(root_path,4,0.2)
test_data_loader = DataLoader(dataset=test_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)


print('===> Begining Training')
def checkpoint(name):
    model_out_path = name
    torch.save(netG.state_dict(), model_out_path)
    print("\n===>Checkpoint saved to {}".format(model_out_path))
    
    
print('===> Building model')
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False).to(device)
netD = define_D(opt.input_nc+4, opt.ndf, 'batch', True).to(device)


# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
criterion= LovaszSoftmax()
#val_criterion=MulticlassDiceLoss()
print('---------- Networks initialized -------------')
print_network(netG)
print('-----------------------------------------------')

def one_hot(label):
    label = label.cpu().numpy()
    one_hot = np.zeros((label.shape[0], 4, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(4):
        one_hot[:,i,...] = (label==i)
    return torch.cuda.FloatTensor(one_hot)

def train(epoch,loss_type=None):
    netG.train()
    
    train_iterator = tqdm(train_data_loader,leave=True, total=len(train_data_loader),position=0,ncols=10)
    iterators=0
    for  inputs, labels in train_iterator:
        iterators=iterators+1
        real_a=inputs.to(device)
        gt_a=labels.to(device)
  
        fake_gt_a = netG(real_a)
        
        optimizerD.zero_grad()
        fake_ab = torch.cat((real_a,fake_gt_a), 1)
        pred_fake = netD.forward(fake_ab.detach())
        
        real_ab = torch.cat((real_a,one_hot(gt_a)), 1)
        pred_real = netD.forward(real_ab.detach())
        
        loss_d_real = torch.mean((pred_real - 1) ** 2)
        loss_d_fake = torch.mean(pred_fake ** 2)
        
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
        optimizerD.step()
        

        optimizerG.zero_grad()
        fake_ab = torch.cat((real_a,fake_gt_a), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan=torch.mean((pred_fake - 1) ** 2)

        loss_g_seg = criterion(fake_gt_a, gt_a.type(torch.cuda.LongTensor)) * opt.lamb
        loss_g = loss_g_gan + loss_g_seg
        loss_g.backward()
        optimizerG.step()
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iterators, len(train_data_loader), loss_d.item(), loss_g.item()))

def test(epoch):
    with torch.no_grad():
 
        netG.eval()
   
        test_epoch_iou= []
        test_iterator = tqdm(test_data_loader,leave=True, total=len(test_data_loader),position=0,ncols=10)
        iterators=0

        for  inputs, labels in test_iterator:
            iterators=iterators+1
            inputs=inputs.to(device)
            labels=labels.to(device)
            y_pred = netG(inputs)
            
            test_iou = mutil_IOU(y_pred, labels)
            test_epoch_iou.append(float(test_iou))
            total_test_iou = np.mean(test_epoch_iou)

            print("===>Epoch[{}]({}/{}): test_iou:{:.4f},avg_test_iou = {:.4f}".format(epoch, iterators, 
                  len(test_data_loader),test_iou, total_test_iou))
    return total_test_iou  

def val(epoch):
    with torch.no_grad():
 
        netG.eval()
   
        val_epoch_iou = []
        val_iterator = tqdm(val_data_loader,leave=True, total=len(val_data_loader),position=0,ncols=10)
        iterators=0

        for  inputs, labels in val_iterator:
            iterators=iterators+1
            inputs=inputs.to(device)
            labels=labels.to(device)
            y_pred = netG(inputs)
            val_iou = mutil_IOU(y_pred, labels)
            val_epoch_iou.append(float(val_iou))
            total_val_iou = np.mean(val_epoch_iou)

            print("===>Epoch[{}]({}/{}): val_iou:{:.4f},avg_val_iou = {:.4f}".format(epoch, iterators, 
                  len(val_data_loader),val_iou, total_val_iou))
            
    return total_val_iou  


if __name__ == '__main__' :
       
        log_dir='checkpoint/model.pth'
        last_log='checkpoint/seq_last.pth'
        best_log='checkpoint/seq_best.pth'
        count=0
        if os.path.exists(log_dir):

            netG.load_state_dict(torch.load(log_dir))
            print('load_weight')
            netG.to(device)  
            print('finetuing model')
            
        for epoch in range(1, opt.nEpochs + 1):    
            train(epoch)
            val_iou=val(epoch)
            test_iou=test(epoch)
            
            if epoch==1:
             
                 test_best_iou=test_iou
                 val_best_iou= val_iou
            
            if val_iou>=val_best_iou:
               checkpoint(log_dir)
            
               test_best_iou= test_iou
               val_best_iou= val_iou
             
            else:
              
                test_best_iou= test_best_iou
                val_best_iou= val_best_iou
                count=count+1
                
            checkpoint(last_log)
            if count>200:
                checkpoint(best_log)
                break

            print("\n===>val_iou not be improved  to {:.4f}".format( val_best_iou))
            print("\n===>test_iou not be improved  to {:.4f}".format( test_best_iou))
            
            

            
