# The base code of this .py file is taken from:
# https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
# , then updated.

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

import os

# parsers from the ViT github repo
parser = argparse.ArgumentParser(description='Active Feature Acquisition Via Explainability-driven Ranking')
parser.add_argument('--dataset', 
                       choices=['cifar10', 'cifar100', 'bloodmnist', 'imagenette'], 
                       default='imagenette', help='Dataset name')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', type=int, default='16')
parser.add_argument('--surr_ckpt')
parser.add_argument('--explainer_ckpt')

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
parser.add_argument('--n_epochs', type=int, default='16')
parser.add_argument('--use_sum', type=int)


parser.add_argument('--init_fea', type=int)
parser.add_argument('--second_fea',  type=int)
parser.add_argument('--third_fea', type=int)

parser.add_argument('--image_size',  type=int)
parser.add_argument('--patch_size',  type=int)

parser.add_argument('--wandb_name', default="afasfasdsdasf")
parser.add_argument('--wandb_project', default="imaginette") #dataset_path
args = parser.parse_args()

def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(os.getenv("CUDA_VISIBLE_DEVICES"))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    os.environ['WANDB_DIR'] = '/project/vkolagrp/osman/wandb'
    #watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project=args.wandb_project,
            name=args.wandb_name)
    wandb.config.update(args)



use_amp = not args.noamp

pretrain_loader, trainloader, valloader, _ = get_dataset(args.dataset)
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
# Loss is CE
criterion = nn.CrossEntropyLoss()




net.load_state_dict(torch.load(args.net_ckpt,map_location=device)) #_layer3_head8_block8_start0
model_gpt.load_state_dict(torch.load(args.GPT_ckpt,map_location=device)) #_start0

if args.shared_backbone == False:
    print('Shared Backbone is FALSE')
    fea_extractor_gpt = deepcopy(net).to(device)
    fea_extractor_gpt.load_state_dict(torch.load(args.GPT_fea_ckpt,map_location=device)) #_start0
    optimizer = optim.Adam(set(list(net.parameters()) + list(model_gpt.parameters())+list(fea_extractor_gpt.parameters())), lr=args.lr) # lr=1e-3 for bs=32  (args.lr)
else:
    print('Shared Backbone is TRUE')
    optimizer = optim.Adam(set(list(net.parameters()) + list(model_gpt.parameters())), lr=args.lr) # lr=1e-3 for bs=32  (args.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
# Load Pretrained FastSHAP model
print('Loading saved surrogate model')
surr = torch.load(args.surr_ckpt).to(device)
surr.eval()
surrogate = ImageSurrogate(surr, width=args.image_size, height=args.image_size, superpixel_size=args.patch_size)
print('Loading saved explainer model')
explainer = torch.load(args.explainer_ckpt).to(device)
explainer.eval()
fastshap = FastSHAP(explainer, surrogate, link=torch.nn.LogSoftmax(dim=1))

# Imputer
imp = ImageImputer(width=args.image_size, height=args.image_size, superpixel_size=args.patch_size)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
net = net.to(device)
soft_func = nn.LogSoftmax(dim=1)
max_float16 = 65504 #3.4028235e+38 #np.finfo(np.float32).max


init_fea = args.init_fea # the first action
second_fea = args.second_fea
third_fea = args.third_fea
def train(epoch):
    print('\nEpoch: %d' % epoch)
    
    net.train()
    model_gpt.train()
    if args.shared_backbone == False:
        fea_extractor_gpt.train()
    train_loss = 0
    correct = 0
    total = 0
    #for iijjkk in range(5):
    iijjkk = 0
    total_loss = 0
    for batch_idx, (inputs1, inputs2, inputs3,  targets) in enumerate(trainloader): #inputs4, inputs5,
        #inputs = inputs.to(device)
        
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)

        values1 = fastshap.shap_values(inputs1) #.to(device)
        values1 = values1[np.arange(values1.shape[0]),np.array(targets),:,:].reshape((values1.shape[0],number_of_actions))

        values2 = fastshap.shap_values(inputs2) #.to(device)
        values2 = values2[np.arange(values1.shape[0]),np.array(targets),:,:].reshape((values1.shape[0],number_of_actions))

        values3 = fastshap.shap_values(inputs3) #.to(device)
        values3 = values3[np.arange(values1.shape[0]),np.array(targets),:,:].reshape((values1.shape[0],number_of_actions))

        values = torch.abs(torch.cat((values1,values2,values3),dim=0))
        

        inputs_original = torch.cat((inputs1,inputs2,inputs3),dim=0)

        targets = targets.to(device)
        targets = torch.cat((targets,targets,targets),dim=0) 
       

        
        S = torch.zeros_like(values,dtype=torch.float) 
        
        ix = torch.randint(max_fea-block_size-1,(S.shape[0],)) 

        max_ix = torch.max(ix)
        sorted_indices_predicted = torch.zeros_like(values)
        sorted_indices_predicted[:,-1] = init_fea
        ####################################################
        input_batch_size = inputs_original.shape[0]

        
        
        inputs_all = torch.repeat_interleave(inputs_original,block_size,dim=0)
        net.eval()
        model_gpt.eval()
        if args.shared_backbone == False:
            fea_extractor_gpt.eval()
        with torch.no_grad():
            S = torch.zeros((input_batch_size,number_of_actions),dtype=torch.float).to(device)
            S[:,init_fea] = 1.0
            S_new = S
            S_reshaped = imp.resize(S)
            #inputs = torch.repeat_interleave(inputs,block_size,dim=0)
            inputs = inputs_original * S_reshaped
           
            targets_gpt = torch.zeros((S.shape[0],1,1)).to(device)
            actions = (torch.ones((S.shape[0],1,1))*init_fea).to(device)
            timesteps = torch.ones((S.shape[0],1,1)).to(device).to(torch.int64)
            count_block = 2
            for ijk in range(max_ix+2):
                
                count_img = 0
                
                
                #outputs= net(inputs)
                outputs, state_embeddings = net(inputs,fea_return=True) #,S_reshaped
                if args.shared_backbone == False:
                    _, state_embeddings = fea_extractor_gpt(inputs,fea_return=True)
                rewards = F.softmax(outputs,dim=1).detach()
                #out_gpt = model_gpt(states = inputs, actions = actions, targets = targets_gpt, rtgs = rewards, timesteps = timesteps)                
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

                sorted_indices_predicted[torch.arange(sorted_indices_predicted.shape[0]),-(ijk+2)] = out_gpt_pred.to(torch.float)
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

                        S_new[torch.arange(count_block-1,S_new.shape[0],count_block),out_gpt_pred] = 1.0 #1,3,5
                     
                        S_reshaped = imp.resize(S_new) 
                        inputs = torch.repeat_interleave(inputs_original,count_block,dim=0)
                        inputs = inputs * S_reshaped                     
                        count_block += 1
                    else:
                        timesteps += 1
                        actions[:,0:(block_size-1),0] = actions[:,1:block_size,0]
                        #print(out_gpt_pred)
                        #print(count_block)
                        actions[torch.arange(input_batch_size),(block_size-1),0]=out_gpt_pred.to(torch.float)
                        #S_new[0:(S_new.shape[0]-1),:] = S_new[1:(S_new.shape[0]),:]
                        S_new = torch.roll(S_new,-1,0)
                        #S_new[0,:] = S_new[1,:]
                        S_new[torch.arange(block_size-1,S_new.shape[0],block_size),:]=S_new[torch.arange(block_size-2,S_new.shape[0],block_size),:]
                        S_new[torch.arange(block_size-1,S_new.shape[0],block_size),out_gpt_pred] = 1.0
                        S_reshaped = imp.resize(S_new)
                        inputs = inputs_all * S_reshaped
                       
        sorted_indices_predicted = sorted_indices_predicted.to(torch.int32)

        net.train()
        model_gpt.train()
        if args.shared_backbone == False:
            fea_extractor_gpt.train()
        ####################################################
        count_img = 0
        S_new = torch.zeros((S.shape[0]*block_size,S.shape[1])).to(device)
        targets_gpt = torch.zeros((S.shape[0],block_size,1)).to(device)
        actions = torch.zeros((S.shape[0],block_size,1)).to(device)
        timesteps = torch.zeros((S.shape[0],1,1)).to(device)
        for i in ix:
            #net.eval()
            #model_gpt.eval()
            for j in range(block_size):
                S_new[block_size*count_img+j, sorted_indices_predicted[count_img,-(i+j+1):].T] = 1.0
                val_tmp = values[count_img,:] - S_new[block_size*count_img+j,:]*1e6
                sorted_indices_tmp = torch.argsort((val_tmp)) # batch, #patch/actions
                targets_gpt[count_img,j,0] = sorted_indices_tmp[-1]
                actions[count_img,j,0] = sorted_indices_predicted[count_img,-(i+j+1)]
            timesteps[count_img,0,0] = i+1
            count_img += 1
      
        S_new = S_new.to(device)
        targets_gpt = targets_gpt.to(device)
        targets_gpt = targets_gpt.to(torch.int64)
        actions = actions.to(device)
        timesteps = timesteps.to(device)
        timesteps = timesteps.to(torch.int64)
        S_reshaped = imp.resize(S_new)

        inputs = inputs_all * S_reshaped
       

        outputs, state_embeddings = net(inputs,fea_return=True) #,S_reshaped
        if args.shared_backbone == False:
            _, state_embeddings = fea_extractor_gpt(inputs,fea_return=True)
        
        loss = (1.0)*criterion(outputs, torch.repeat_interleave(targets,block_size,dim=0))

        rewards = F.softmax(outputs,dim=1).detach()
         
        out_gpt = model_gpt(state_embeddings = state_embeddings, actions = actions, targets = targets_gpt, rtgs = rewards, timesteps = timesteps)
        
        out_gpt = out_gpt.reshape(-1, out_gpt.size(-1))
        out_gpt = out_gpt - S_new*1e6
        loss += (1.0)*F.cross_entropy(out_gpt, targets_gpt.reshape(-1))
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.cpu().item()
        #if iijjkk == 0:
        #    break
    print(total_loss)
    return loss

##### Validation
def test(epoch):
    #global best_acc
    net.eval()
    model_gpt.eval()
    if args.shared_backbone == False:
        fea_extractor_gpt.eval()
    test_loss = 0
    correct = torch.zeros(max_val_fea)
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs_original, targets) in enumerate(valloader):
                input_batch_size = inputs_original.shape[0]

                total += input_batch_size
                inputs_original  = inputs_original.to(device)

                inputs_all = torch.repeat_interleave(inputs_original,block_size,dim=0)
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
                for ijk in range(max_val_fea):
                    
                    count_img = 0
                    

                    outputs, state_embeddings = net(inputs,fea_return=True) #,S_reshaped
                    if args.shared_backbone == False:
                        _, state_embeddings = fea_extractor_gpt(inputs,fea_return=True)
                    
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
                            S_new[torch.arange(count_block-1,S_new.shape[0],count_block),out_gpt_pred] = 1.0 
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
                            S_new[torch.arange(block_size-1,S_new.shape[0],block_size),out_gpt_pred] = 1.0
                            S_reshaped = imp.resize(S_new)
                            inputs = inputs_all * S_reshaped
                 
                    _,predicted = torch.max(outputs, 1)
                    
                    if ijk >= (block_size-1):
                        predicted=predicted[torch.arange((block_size-1),predicted.shape[0],block_size)] 
                    else:
                        predicted=predicted[torch.arange(count_block-3,predicted.shape[0],count_block-2)] 
                   
                    correct[ijk] += predicted.eq(targets).sum().item()

    
    correct = torch.mean(correct).item()
    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
#net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_period = 2
    if (epoch)%val_period==0 or (epoch==(args.n_epochs-1)):
        val_loss, acc = test(epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),args.net_ckpt.replace('.ckpt', '_second_stage.ckpt')) 
            torch.save(model_gpt.state_dict(),args.GPT_ckpt.replace('.ckpt', '_second_stage.ckpt'))
            if args.shared_backbone == False:
                torch.save(fea_extractor_gpt.state_dict(),args.GPT_fea_ckpt.replace('.ckpt', '_second_stage.ckpt'))
        
    
    scheduler.step(epoch-1) 
    
    # Log training..
    if (epoch)%val_period==0 or (epoch==(args.n_epochs-1)) :
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})
    else:
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})
    if epoch==(args.n_epochs-1):
        wandb.finish()


    
