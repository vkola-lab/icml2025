from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

from tabular_dataset import get_dataset, data_split
import os
from model_gpt import GPT, GPTConfig

import argparse
import pandas as pd
import csv
import time

from torchmetrics import  Accuracy, AUROC
# FastSHAP related
#from resnet import ResNet18, ResNet34
from copy import deepcopy
from torch.utils.data import Dataset
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Active Feature Acquisition Via Explainability-driven Ranking')

parser.add_argument('--dataset', 
                       choices=['spam', 'metabric', 'cps', 'ctgs', 'ckd'], 
                       default='metabric', help='Dataset name')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')

parser.add_argument('--net_ckpt')
parser.add_argument('--GPT_ckpt')

parser.add_argument('--n_blocks', type=int, default='8')
parser.add_argument('--n_layer', type=int, default='3')
parser.add_argument('--n_head', type=int, default='8') #num_classes

parser.add_argument('--n_hidden',  type=int)
parser.add_argument('--n_embd',  type=int)

parser.add_argument('--max_fea', type=int, default='50')
parser.add_argument('--max_val_fea', type=int, default='20')

parser.add_argument('--n_epochs', type=int, default='200')

parser.add_argument('--init_fea',  type=int)
parser.add_argument('--second_fea',  type=int)
parser.add_argument('--third_fea',  type=int)
args = parser.parse_args()

class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.
    
    Args:
      append:
      mask_size:
    '''
    def __init__(self, append, mask_size=None):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, m):
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out

class MLP_tabular(nn.Module):
    def __init__(self,d_in,d_out,n_hidden = 128, n_embd = 512):
        super().__init__()        
             
        self.lin1 = nn.Linear(d_in, n_hidden)           
        self.lin2 = nn.Linear(n_hidden, n_hidden)          
        self.lin3 = nn.Linear(n_hidden, d_out)

        self.lin3_fea = nn.Linear(n_hidden, int(n_hidden/2))
        self.lin4_fea = nn.Linear(int(n_hidden/2), n_embd)

    def forward(self, x_in, fea_return=False):
        x = self.lin1(x_in)
        fea = self.lin2(F.dropout(F.relu((x)),p=0.3, training=self.training))
        x = self.lin3(F.dropout(F.relu((fea)),p=0.3, training=self.training))

        if fea_return:
            fea = self.lin3_fea(F.dropout(F.relu((fea)),p=0.3, training=self.training))
            fea = self.lin4_fea(F.dropout(F.relu((fea)),p=0.3, training=self.training))
            fea = F.tanh(fea)
            return x,fea
        else:
            return x



np.random.seed(0)

dataset = get_dataset(args.dataset)

d_in = dataset.input_size
d_out = dataset.output_size


if d_out == 2:
    metric = lambda pred, y: AUROC(num_classes=d_out)(pred.softmax(dim=1), y) 
else:
    metric = Accuracy(num_classes=d_out) 

mean = dataset.tensors[0].mean(dim=0)
std = torch.clamp(dataset.tensors[0].std(dim=0), min=1e-3)
dataset.tensors = ( (dataset.tensors[0] - mean)/ std, dataset.tensors[1], dataset.tensors[2])
train_dataset, val_dataset, test_dataset = data_split(dataset) #, if_metabric=True


testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)


init_fea=args.init_fea
second_fea=args.second_fea
third_fea=args.third_fea

net = MLP_tabular(d_in=2*d_in,d_out=d_out, n_hidden=args.n_hidden, n_embd=args.n_embd) 
net = net.to(device)



number_of_actions = d_in #vocab_size
block_size = args.n_blocks
nhead = args.n_head
max_fea = args.max_fea 
max_val_fea = args.max_val_fea


mconf = GPTConfig(number_of_actions,  args.n_blocks,
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, model_type='reward_conditioned', max_timestep=number_of_actions)
model_gpt = GPT(mconf,d_out) #conf,state_emb_net,n_class
model_gpt = model_gpt.to(device)

mask_layer = MaskLayer(append=True)


net.load_state_dict(torch.load(args.net_ckpt,map_location=device)) #_layer3_head8_block8_start0 _second_stage _second_stage
model_gpt.load_state_dict(torch.load(args.GPT_ckpt,map_location=device)) #_start0 _second_stage



net.eval()
model_gpt.eval()
#net_gpt.eval()
test_loss = 0


total = 0
pred_list = [[] for _ in range(max_val_fea)]
label_list = []


with torch.no_grad():
    for batch_idx, (inputs_original,_, targets) in enumerate(testloader):
            label_list.append(targets)
            input_batch_size = inputs_original.shape[0]

            total += input_batch_size
            inputs_original  = inputs_original.to(device)

            inputs_all = torch.repeat_interleave(inputs_original,block_size,dim=0)
            targets = targets.to(device)
                        
            S = torch.zeros((input_batch_size,number_of_actions),dtype=torch.float).to(device)
            S[:,init_fea] = 1.0
            S_new = S
            inputs = mask_layer(inputs_original, S_new)
            targets_gpt = torch.zeros((S.shape[0],1,1)).to(device)
            actions = (torch.ones((S.shape[0],1,1))*init_fea).to(device)
            timesteps = torch.ones((S.shape[0],1,1)).to(device).to(torch.int64)
            count_block = 2
            for ijk in range(max_val_fea): 
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

                if actions.shape[1] < block_size:
                    actions_to_add = torch.ones((input_batch_size,1,1),device=device)
                    actions_to_add[torch.arange(input_batch_size),0,0] = out_gpt_pred.to(torch.float)
                    
                    actions = torch.cat((actions,actions_to_add),dim=1)
                    S_old = S_new
                    S_new = torch.repeat_interleave(S,count_block,dim=0)

                    for ijk_tmp in range(1,count_block-1):
                        S_new[torch.arange(ijk_tmp,S_new.shape[0],count_block),:] = S_old[torch.arange(ijk_tmp,S_old.shape[0],count_block-1),:]
                    
                    S_new[torch.arange(count_block-1,S_new.shape[0],count_block),:] = S_new[torch.arange(count_block-2,S_new.shape[0],count_block),:]

                    S_new[torch.arange(count_block-1,S_new.shape[0],count_block),out_gpt_pred] = 1.0 #1,3,5


                    inputs = torch.repeat_interleave(inputs_original,count_block,dim=0)
                    inputs = mask_layer(inputs, S_new)

                    count_block += 1
                else:
                    timesteps += 1
                    actions[:,0:(block_size-1),0] = actions[:,1:block_size,0]
                    actions[torch.arange(input_batch_size),(block_size-1),0]=out_gpt_pred.to(torch.float)
                    S_new = torch.roll(S_new,-1,0)
                    S_new[torch.arange(block_size-1,S_new.shape[0],block_size),:]=S_new[torch.arange(block_size-2,S_new.shape[0],block_size),:]
                    S_new[torch.arange(block_size-1,S_new.shape[0],block_size),out_gpt_pred] = 1.0
                    inputs = mask_layer(inputs_all, S_new)

                _,predicted = torch.max(outputs, 1)
                
                if ijk >= (block_size-1):
                    predicted=predicted[torch.arange((block_size-1),predicted.shape[0],block_size)] #count_block-2::count_block-1
                    out_append = outputs[torch.arange((block_size-1),outputs.shape[0],block_size),:].cpu()
                    pred_list[ijk].append(out_append)
                else:
                    predicted=predicted[torch.arange(count_block-3,predicted.shape[0],count_block-2)] #count_block-2::count_block-1
                    out_append = outputs[torch.arange(count_block-3,outputs.shape[0],count_block-2),:].cpu()
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

