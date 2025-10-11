from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from model_gpt import *

# FastSHAP related
from fastshap import FastSHAP
from fastshap import ImageSurrogate
from fastshap.image_imputers import ImageImputer

from copy import deepcopy
from torch.utils.data import Dataset
from fastshap import ImageSurrogate
from unet import UNet
from fastshap import FastSHAP
from torchvision.datasets import ImageFolder
from fastai.vision.all import untar_data, URLs
from image_transforms import get_dataset
from image_models import get_model

from torchmetrics import  Accuracy, AUROC
import os

print(os.getenv("CUDA_VISIBLE_DEVICES"))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# parsers from the ViT github repo
parser = argparse.ArgumentParser(description='Active Feature Acquisition Via Explainability-driven Ranking')
parser.add_argument('--dataset', 
                       choices=['cifar10', 'cifar100', 'bloodmnist', 'imagenette'], 
                       default='imagenette', help='Dataset name')

parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', type=int, default='16')
#parser.add_argument('--surr_ckpt')
#parser.add_argument('--explainer_ckpt')

parser.add_argument('--net_ckpt')
parser.add_argument('--GPT_ckpt')
parser.add_argument('--GPT_fea_ckpt')

parser.add_argument('--pretrained_model_name', default="vit_small_patch16_224")
parser.add_argument('--n_blocks', type=int, default='4')
parser.add_argument('--n_layer', type=int, default='3')
parser.add_argument('--n_head', type=int, default='4')
parser.add_argument('--num_classes', type=int, default='10')
parser.add_argument('--max_fea', type=int, default='50')
parser.add_argument('--max_val_fea', type=int, default='20')

parser.add_argument('--shared_backbone', action='store_true', help="Enable the feature")

parser.add_argument('--use_sum', type=int)


parser.add_argument('--init_fea', type=int)
parser.add_argument('--second_fea',  type=int)
parser.add_argument('--third_fea', type=int) 

parser.add_argument('--image_size',  type=int)
parser.add_argument('--patch_size',  type=int)

args = parser.parse_args()


_, _, _, testloader = get_dataset(args.dataset)
net = get_model(args.dataset, num_classes=args.num_classes).to(device)

# Set up mini-GPT model
number_of_actions = int((args.image_size//args.patch_size)**2)#vocab_size
block_size = args.n_blocks
nhead = args.n_head 
max_fea = args.max_fea 
max_val_fea = args.max_val_fea

# Create dummy input to get feature dimension
dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
with torch.no_grad():
    _, fea = net(dummy_input, fea_return=True)

if len(fea.shape) >= 4:
    n_embd = fea.shape[1]*fea.shape[2]*fea.shape[3]
else:
    n_embd = fea.shape[1]  

mconf = GPTConfig(number_of_actions, block_size,
                  n_layer=args.n_layer, n_head=nhead, n_embd=n_embd, model_type='reward_conditioned', max_timestep=number_of_actions)
model_gpt = GPT(mconf,args.num_classes) #conf,state_emb_net,n_class
model_gpt = model_gpt.to(device)


net.load_state_dict(torch.load(args.net_ckpt,map_location=device)) #_layer3_head8_block8_start0
model_gpt.load_state_dict(torch.load(args.GPT_ckpt,map_location=device)) #_start0


imp = ImageImputer(width=args.image_size, height=args.image_size, superpixel_size=args.patch_size)

init_fea = args.init_fea # the first action
second_fea = args.second_fea
third_fea = args.third_fea


if args.num_classes == 2:
    metric = lambda pred, y: AUROC(num_classes=args.num_classes)(pred.softmax(dim=1), y) 
else:
    metric = Accuracy(num_classes=args.num_classes) 

total = 0
net.eval()
model_gpt.eval()

pred_list = [[] for _ in range(max_val_fea)]
label_list = []
with torch.no_grad():
    for batch_idx, (inputs_original, targets) in enumerate(testloader):
            label_list.append(targets)
            input_batch_size = inputs_original.shape[0]

            total += input_batch_size
            inputs_original  = inputs_original.to(device)


            targets = targets.to(device)

            S = torch.zeros((input_batch_size,number_of_actions),dtype=torch.float).to(device)
            S[:,init_fea] = 1.0
            S_new = S
            S_reshaped = imp.resize(S)
            inputs = inputs_original * S_reshaped

            targets_gpt = torch.zeros((S.shape[0],1,1)).to(device)
            actions = (torch.ones((S.shape[0],1,1))*init_fea).to(device)
            timesteps = torch.ones((S.shape[0],1,1)).to(device).to(torch.int64)
            count_block = 2

            inputs_all = torch.repeat_interleave(inputs_original,block_size,dim=0)
            for ijk in range(max_val_fea):
                count_img = 0
                outputs, state_embeddings = net(inputs,fea_return=True)

                
                rewards = F.softmax(outputs,dim=1).detach()
                out_gpt = model_gpt(state_embeddings = state_embeddings, actions = actions, targets = targets_gpt, rtgs = rewards, timesteps = timesteps)
                out_gpt = out_gpt.reshape(-1, out_gpt.size(-1))
                out_gpt = out_gpt - S_new*1e6


                out_gpt = out_gpt[torch.arange(count_block-2,out_gpt.shape[0],count_block-1)]
                _, out_gpt_pred = torch.max(out_gpt, 1)
                if ijk == 0:
                        out_gpt_pred = torch.ones_like(out_gpt_pred,device=device)
                        out_gpt_pred = out_gpt_pred*second_fea
                if ijk == 1:
                        out_gpt_pred = torch.ones_like(out_gpt_pred,device=device)
                        out_gpt_pred = out_gpt_pred*third_fea

                if block_size==1:
                        timesteps += 1
                        actions[torch.arange(input_batch_size),(block_size-1),0]=out_gpt_pred.to(torch.float)                        
                        S_new[torch.arange(block_size-1,S_new.shape[0],block_size),out_gpt_pred] = 1.0
                        S_reshaped = imp.resize(S_new)
                        inputs = inputs_all * S_reshaped
       
                else:
                    if actions.shape[1] < block_size:
                        actions_to_add = torch.ones((input_batch_size,1,1),device=device)
                        actions_to_add[torch.arange(input_batch_size),0,0] = out_gpt_pred.to(torch.float)
                        
                        actions = torch.cat((actions,actions_to_add),dim=1)
                        S_old = S_new
                        S_new = torch.repeat_interleave(S,count_block,dim=0)
                        
                        for ijk_tmp in range(1,count_block-1):
                            S_new[torch.arange(ijk_tmp,S_new.shape[0],count_block),:] = S_old[torch.arange(ijk_tmp,S_old.shape[0],count_block-1),:]
                        
                        S_new[torch.arange(count_block-1,S_new.shape[0],count_block),:] = S_new[torch.arange(count_block-2,S_new.shape[0],count_block),:]

                        S_new_reject = S_new[torch.arange(count_block-1,S_new.shape[0],count_block),:]
                        S_new_reject = imp.resize(S_new_reject)

                        S_new[torch.arange(count_block-1,S_new.shape[0],count_block),out_gpt_pred] = 1.0 #1,3,5
                        
                        S_reshaped = imp.resize(S_new)
                    
                        inputs = torch.repeat_interleave(inputs_original,count_block,dim=0)
                   
                        inputs = inputs * S_reshaped
              
        
                        count_block += 1
                    else:
                        timesteps += 1
                        actions[:,0:(block_size-1),0] = actions[:,1:block_size,0]
                        actions[torch.arange(input_batch_size),(block_size-1),0]=out_gpt_pred.to(torch.float)
                    
                        S_new = torch.roll(S_new,-1,0)
                       
                        S_new[torch.arange(block_size-1,S_new.shape[0],block_size),:]=S_new[torch.arange(block_size-2,S_new.shape[0],block_size),:]
                       
                       
                        S_new_reject = S_new[torch.arange(block_size-1,S_new.shape[0],block_size),:]
                        S_new_reject = imp.resize(S_new_reject)

                        S_new[torch.arange(block_size-1,S_new.shape[0],block_size),out_gpt_pred] = 1.0
                        S_reshaped = imp.resize(S_new)
                        inputs = inputs_all * S_reshaped

                _,predicted = torch.max(F.softmax(outputs,dim=1), 1)
                
                if ijk >= (block_size-1):
                    predicted=predicted[torch.arange((block_size-1),predicted.shape[0],block_size)]
                    out_append = outputs[torch.arange((block_size-1),outputs.shape[0],block_size),:].cpu()
                    pred_list[ijk].append(out_append)
                else:
                    predicted=predicted[torch.arange(count_block-3,predicted.shape[0],count_block-2)]
                    out_append = outputs[torch.arange(count_block-3,outputs.shape[0],count_block-2)].cpu()
                    pred_list[ijk].append(out_append)

               
list_of_metric=[]
metric_average = 0

for i in range(max_val_fea):
    pred = torch.cat(pred_list[i], 0)
    y = torch.cat(label_list, 0)
    list_of_metric.append(metric(pred, y))
    metric_average += metric(pred, y)
    
print(list_of_metric)
print('Average:')
print(metric_average/max_val_fea)

