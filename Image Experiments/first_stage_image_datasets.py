# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision

import argparse
import pandas as pd
import csv
import time
# "vit_small_patch16_224"
from models import *
#from resnet import ResNet18, ResNet34
from copy import deepcopy
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# FastSHAP related
from fastshap import FastSHAP
from fastshap import ImageSurrogate
from fastshap.utils import MaskLayer2d, MaskLayer2dPre, KLDivLoss, DatasetInputOnly
from fastshap.image_imputers import ImageImputer
from image_models import get_model
from fastshap import ImageSurrogate
from fastshap import FastSHAP


from image_transforms import get_dataset
from image_models import get_model
# Set the device
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.getenv("CUDA_VISIBLE_DEVICES"))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# parsers from the ViT github repo
parser = argparse.ArgumentParser(description='Active Feature Acquisition Via Explainability-driven Ranking')

parser.add_argument('--dataset', 
                       choices=['cifar10', 'cifar100', 'bloodmnist', 'imagenette'], 
                       default='imagenette', help='Dataset name')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--pretrain_lr', default=1e-3, type=float, help='pretrain learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--surr_ckpt')
parser.add_argument('--explainer_ckpt')

parser.add_argument('--net_ckpt')
parser.add_argument('--GPT_ckpt')
parser.add_argument('--GPT_fea_ckpt')
parser.add_argument('--n_blocks', type=int, default='4')
parser.add_argument('--n_layer', type=int, default='3')
parser.add_argument('--n_convs', type=int, default='2')
parser.add_argument('--n_head', type=int, default='4') #num_classes
parser.add_argument('--num_classes', type=int, default='10')

parser.add_argument('--max_fea', type=int, default='50')
parser.add_argument('--max_val_fea', type=int, default='20')

parser.add_argument('--shared_backbone', action='store_true', help="Enable the feature")
parser.add_argument('--n_epochs', type=int, default='200')

parser.add_argument('--init_fea',  type=int)
parser.add_argument('--second_fea',  type=int)
parser.add_argument('--third_fea',  type=int)

parser.add_argument('--image_size',  type=int)
parser.add_argument('--patch_size',  type=int)

parser.add_argument('--wandb_name') #dataset_path
parser.add_argument('--wandb_project') #dataset_path
parser.add_argument('--dataset_path')
args = parser.parse_args()

# Initialize wandb 
usewandb = ~args.nowandb
if usewandb:
    import wandb
    os.environ['WANDB_DIR'] = '/project/vkolagrp/osman/wandb'
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project=args.wandb_project,
            name=args.wandb_name) # it was 32
    wandb.config.update(args)


# Function to pretrain the model (first stage of the training) taken from the Greedy paper's code
def generate_uniform_mask(batch_size, num_features): # Generating random missingness; used in the pretraining/first-stage
    '''Generate binary masks with cardinality chosen uniformly at random.'''
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()

def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param

def move_features_to_end(sorted_indices, features_to_move, device):
    """
    Move the specified features to the end of the tensor.

    Parameters:
    sorted_indices (torch.Tensor): The tensor with sorted indices.
    features_to_move (list or torch.Tensor): The features to move to the end.
    device (torch.device): The device on which tensors are located.

    Returns:
    torch.Tensor: The tensor with specified features moved to the end.
    """
    # Ensure features_to_move is a torch tensor
    features_to_move = torch.tensor(features_to_move, dtype=sorted_indices.dtype, device=device)
    
    # Create a mask for the features to move
    mask = torch.isin(sorted_indices, features_to_move)
    
    # Select the rows excluding the features to move
    tensor_without_the_elements = sorted_indices[~mask].view(sorted_indices.size(0), -1)
    
    # Repeat features_to_move tensor to match batch size
    repeated_features_to_move = features_to_move.repeat(sorted_indices.size(0), 1)
    
    # Create a new tensor with the reordered rows
    sorted_indices = torch.cat([
        tensor_without_the_elements, 
        repeated_features_to_move
    ], dim=1)
    
    return sorted_indices


class MaskingPretrainer(nn.Module):
    '''Pretrain model with missing features.'''

    def __init__(self, model, mask_layer):
        super().__init__()
        self.model = model
        self.mask_layer = mask_layer
        
    def fit(self,
            train_loader,
            val_loader,
            lr,
            nepochs,
            loss_fn,
            val_loss_fn=None,
            val_loss_mode=None,
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            early_stopping_epochs=None,
            verbose=True):
        '''
        Train model.
        
        Args:
          train_loader:
          val_loader:
          lr:
          nepochs:
          loss_fn:
          val_loss_fn:
          val_loss_mode:
          factor:
          patience:
          min_lr:
          early_stopping_epochs:
          verbose:
        '''
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')
        
        # Set up optimizer and lr scheduler.
        model = self.model
        mask_layer = self.mask_layer
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose)
        
        # Determine mask size.
        mask_size = int((args.image_size//args.patch_size)**2)
        imp = ImageImputer(width=args.image_size, height=args.image_size, superpixel_size=args.patch_size)
        
        # For tracking best model and early stopping.
        best_model = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
            
        for epoch in range(nepochs):
            # Switch model to training mode.
            model.train()

            for x, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)
                
                # Generate missingness.
                m = generate_uniform_mask(len(x), mask_size).to(device)
                m = imp.resize(m)
                # Calculate loss.
                x_masked = mask_layer(x, m)
                pred = model(x_masked)
                loss = loss_fn(pred, y)

                # Take gradient step.
                loss.backward()
                opt.step()
                model.zero_grad()
                
            # Calculate validation loss.
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for x, y in val_loader:
                    # Move to device.
                    x = x.to(device)
                    
                    # Generate missingness.
                    # TODO this should be precomputed and shared across epochs
                    m = generate_uniform_mask(len(x), mask_size).to(device)
                    m = imp.resize(m)
                    # Calculate prediction.
                    x_masked = mask_layer(x, m)
                    pred = model(x_masked)
                    pred_list.append(pred.cpu())
                    label_list.append(y.cpu())
                    
                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()
                
            
            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Val loss = {val_loss:.4f}\n')
                
            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_model = deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                
            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f'Stopping early at epoch {epoch+1}')
                break

        # Copy parameters from best model.
        restore_parameters(model, best_model)



pretrain_loader, trainloader, valloader, _ = get_dataset(args.dataset)

# Prepare datasets

# Load Pretrained FastSHAP model
print('Loading saved surrogate model')
#surr = torch.load('notebooks/cifar surrogate_size4.pt').to(device)
surr = torch.load(args.surr_ckpt).to(device)
surr.eval()
surrogate = ImageSurrogate(surr, width=args.image_size, height=args.image_size, superpixel_size=args.patch_size)
print('Loading saved explainer model')
explainer = torch.load(args.explainer_ckpt).to(device)
explainer.eval()
fastshap = FastSHAP(explainer, surrogate, link=torch.nn.LogSoftmax(dim=1))

# Imputer
imp = ImageImputer(width=args.image_size, height=args.image_size, superpixel_size=args.patch_size)




net = get_model(args.dataset, num_classes=args.num_classes).to(device)

# Set up mini-GPT model
number_of_features = int((args.image_size//args.patch_size)**2)
block_size = args.n_blocks
nhead = args.n_head #8 # it was 8
max_fea = args.max_fea # it was 32
max_val_fea = args.max_val_fea

# Create dummy input to get feature dimension
dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
with torch.no_grad():
    _, fea = net(dummy_input, fea_return=True)
# Get feature dimension from network output
if len(fea.shape) >= 4:
    n_embd = fea.shape[1]*fea.shape[2]*fea.shape[3]
else:
    n_embd = fea.shape[1]  

#print(fea.shape)
#import time; time.sleep(100)

mconf = GPTConfig(number_of_features, block_size,
                  n_layer=args.n_layer, n_head=nhead, n_embd=n_embd, model_type='reward_conditioned', max_timestep=number_of_features)
model_gpt = GPT(mconf,args.num_classes) #conf,state_emb_net,n_class
model_gpt = model_gpt.to(device)
# Loss is CE
criterion = nn.CrossEntropyLoss()



mask_layer = MaskLayer2dPre(append=False,value=0)
pretrain = MaskingPretrainer(net, mask_layer).to(device)

print('beginning pre-training...')
pretrain.fit(
    pretrain_loader,
    valloader,
    lr=args.pretrain_lr,
    nepochs=100,
    loss_fn=nn.CrossEntropyLoss(),
    verbose=True)
print('done pretraining')

if args.shared_backbone == False:
    print('Shared Backbone is FALSE')
    fea_extractor_gpt = deepcopy(net).to(device)
    optimizer = optim.Adam(set(list(net.parameters()) + list(model_gpt.parameters())+list(fea_extractor_gpt.parameters())), lr=args.lr) # lr=1e-3 for bs=32  (args.lr)
else:
    print('Shared Backbone is TRUE')
    optimizer = optim.Adam(set(list(net.parameters()) + list(model_gpt.parameters())), lr=args.lr) 

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)



##### Training
use_amp = not args.noamp
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
soft_func = nn.LogSoftmax(dim=1)
max_float16 = 65504 #3.4028235e+38 #np.finfo(np.float32).max

init_fea = args.init_fea # the first action
second_fea = args.second_fea
third_fea = args.third_fea
# The prediction network is trained in a classical way
# The mini-GPT model is trained as the next token predcition, i.e. next action prediction


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
    for batch_idx, (inputs1, inputs2, inputs3, targets) in enumerate(trainloader): #inputs4, inputs5,
        
        
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        inputs3 = inputs3.to(device)

        values1 = fastshap.shap_values(inputs1) #.to(device)
        values1 = values1[np.arange(values1.shape[0]),np.array(targets),:,:].reshape((values1.shape[0],number_of_features))

        values2 = fastshap.shap_values(inputs2) #.to(device)
        values2 = values2[np.arange(values1.shape[0]),np.array(targets),:,:].reshape((values1.shape[0],number_of_features))

        values3 = fastshap.shap_values(inputs3) #.to(device)
        values3 = values3[np.arange(values1.shape[0]),np.array(targets),:,:].reshape((values1.shape[0],number_of_features))

        values = torch.abs(values1) + torch.abs(values2) + torch.abs(values3) # diff sum
        #values = torch.cat((values1,values2,values3),dim=0)
        sorted_indices = torch.argsort((values), dim=1) # batch, #patch/actions
        sorted_indices = torch.cat((sorted_indices,sorted_indices,sorted_indices),dim=0) 

        #values = torch.abs(torch.cat((values1,values2,values3),dim=0))

        #sorted_indices = torch.argsort((values), dim=1) # batch, #patch/actions
        
        inputs = torch.cat((inputs1,inputs2,inputs3),dim=0) #,inputs4,inputs5

        targets = targets.to(device)
        targets = torch.cat((targets,targets,targets),dim=0) 

        sorted_indices = (move_features_to_end(sorted_indices,[third_fea,second_fea,init_fea],device))

        S = torch.zeros_like(sorted_indices,dtype=torch.float) #.to(device)
        
        ix = torch.randint(max_fea-block_size-1,(S.shape[0],)) # 20 is # of max features

        S_new = torch.zeros((S.shape[0]*block_size,S.shape[1]))
        targets_gpt = torch.zeros((S.shape[0],block_size,1))
        actions = torch.zeros((S.shape[0],block_size,1))
        timesteps = torch.zeros((S.shape[0],1,1))
        count_img = 0
        
        #S_new = torch.stack([S])
        for i in ix:
            for j in range(block_size):
                S_new[block_size*count_img+j, sorted_indices[count_img,-(i+j+1):].T] = 1.0
                targets_gpt[count_img,j,0] = sorted_indices[count_img,-(i+j+2)]
                actions[count_img,j,0] = sorted_indices[count_img,-(i+j+1)]
            timesteps[count_img,0,0] = i+1
            count_img += 1
        
        S_new = S_new.to(device)
        targets_gpt = targets_gpt.to(device)
        targets_gpt = targets_gpt.to(torch.int64)
        actions = actions.to(device)
        timesteps = timesteps.to(device)
        timesteps = timesteps.to(torch.int64)
        S_reshaped = imp.resize(S_new)
        inputs = torch.repeat_interleave(inputs,block_size,dim=0)
        inputs = inputs * S_reshaped


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
    correct = torch.zeros(max_val_fea) #instead of max_fea there was 30
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs_original, targets) in enumerate(valloader):
                input_batch_size = inputs_original.shape[0]

                total += input_batch_size
                inputs_original  = inputs_original.to(device)

                inputs_all = torch.repeat_interleave(inputs_original,block_size,dim=0)
                targets = targets.to(device)

                S = torch.zeros((input_batch_size,number_of_features),dtype=torch.float).to(device)
                S[:,init_fea] = 1.0
                S_new = S
                S_reshaped = imp.resize(S)
            
                inputs = inputs_original * S_reshaped

                targets_gpt = torch.zeros((S.shape[0],1,1)).to(device)
                actions = (torch.ones((S.shape[0],1,1))*init_fea).to(device)
                timesteps = torch.ones((S.shape[0],1,1)).to(device).to(torch.int64)
                count_block = 2
                for ijk in range(max_val_fea): #instead of max_fea there was 30
                    
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

                    # The indices of the first three features/actions are fixed.
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
                            actions[:,0:(block_size-1),0] = actions[:,1:block_size,0].clone()
                            
                            actions[torch.arange(input_batch_size),(block_size-1),0]=out_gpt_pred.to(torch.float)
                            
                            S_new = torch.roll(S_new,-1,0)
                            
                            S_new[torch.arange(block_size-1,S_new.shape[0],block_size),:]=S_new[torch.arange(block_size-2,S_new.shape[0],block_size),:]
                            S_new[torch.arange(block_size-1,S_new.shape[0],block_size),out_gpt_pred] = 1.0
                            S_reshaped = imp.resize(S_new)
                            inputs = inputs_all * S_reshaped
                           

                    _,predicted = torch.max(outputs, 1)
                    
                    if ijk >= (block_size-1):
                        predicted=predicted[torch.arange((block_size-1),predicted.shape[0],block_size)] #count_block-2::count_block-1
                    else:
                        predicted=predicted[torch.arange(count_block-3,predicted.shape[0],count_block-2)] #count_block-2::count_block-1
                   
                    correct[ijk] += predicted.eq(targets).sum().item()

    
    correct = torch.mean(correct).item()
    acc = 100.*correct/total
    print(acc)
    return test_loss, acc

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch    

# Call training function for each epoch and call validation function for each 5 epochs
# I call this training stage as the second stage
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    if (epoch)%5==0 or (epoch==(args.n_epochs-1)):
        val_loss, acc = test(epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),args.net_ckpt) #vit_init_fea_134_lr_1e5_nconvs8 with bnorm
            torch.save(model_gpt.state_dict(),args.GPT_ckpt )
            if args.shared_backbone == False:
                torch.save(fea_extractor_gpt.state_dict(),args.GPT_fea_ckpt)
    
    scheduler.step(epoch-1) # step cosine scheduling      

    
    # Log training..
    if (epoch)%5==0 or (epoch==(args.n_epochs-1)) :
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})
    else:
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})
 


    
