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

from models import *

from copy import deepcopy
from torch.utils.data import Dataset
from fastshap.utils import MaskLayer

from tabular_dataset import get_dataset, data_split
from torchmetrics import  Accuracy, AUROC

from tabular_dataset import get_dataset, data_split
import os
from gpt_model import GPT, GPTConfig
print(os.getenv("CUDA_VISIBLE_DEVICES"))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from torch.utils.data import Dataset
from torchmetrics import  Accuracy, AUROC

# parsers
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
parser.add_argument('--GPT_fea_ckpt')
parser.add_argument('--n_blocks', type=int, default='8')
parser.add_argument('--n_layer', type=int, default='3')
parser.add_argument('--n_convs', type=int, default='2')
parser.add_argument('--n_head', type=int, default='8') #num_classes

parser.add_argument('--n_hidden',  type=int)
parser.add_argument('--n_embd',  type=int)

parser.add_argument('--max_fea', type=int, default='50')
parser.add_argument('--max_val_fea', type=int, default='20')

parser.add_argument('--shared_backbone', action='store_true', help="Enable the feature")
parser.add_argument('--n_epochs', type=int, default='16')

parser.add_argument('--init_fea',  type=int)
parser.add_argument('--second_fea',  type=int)
parser.add_argument('--third_fea',  type=int)

parser.add_argument('--wandb_name') 
parser.add_argument('--wandb_project') 
parser.add_argument('--dataset_path')
args = parser.parse_args()


def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param


# take in args
usewandb = ~args.nowandb
use_amp = not args.noamp


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

class TabularSHAP(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset        
        #self.shap_vals = shap_vals
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, x_shap, y = self.dataset[idx]
        #x_shap = self.shap_vals[idx,:,:]
              
        return x, x_shap, y

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



# Initialize wandb 
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project=args.wandb_project,
            name=args.wandb_name) # it was 32
    wandb.config.update(args)


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

init_fea=args.init_fea
second_fea=args.second_fea
third_fea=args.third_fea

train_dataset = TabularSHAP(train_dataset)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8) #bs #32

valloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

net = MLP_tabular(d_in=2*d_in,d_out=d_out, n_hidden=args.n_hidden, n_embd=args.n_embd) 
net = net.to(device)



number_of_actions = d_in #vocab_size
block_size = args.n_blocks
nhead = args.n_head

max_fea = args.max_fea 
max_val_fea = args.max_val_fea

mconf = GPTConfig(number_of_actions, args.n_blocks,
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, model_type='reward_conditioned', max_timestep=number_of_actions)
model_gpt = GPT(mconf,d_out) #conf,state_emb_net,n_class
model_gpt = model_gpt.to(device)
# Loss is CE
criterion = nn.CrossEntropyLoss()

mask_layer = MaskLayer(append=True)





net.load_state_dict(torch.load(args.net_ckpt)) #_layer3_head8_block8_start0
model_gpt.load_state_dict(torch.load(args.GPT_ckpt)) #_start0

optimizer = optim.Adam(set(list(net.parameters()) +list(model_gpt.parameters())), lr=args.lr, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
net = net.to(device)
soft_func = nn.LogSoftmax(dim=1)
max_float16 = 65504 #3.4028235e+38 #np.finfo(np.float32).max


def train(epoch):
    print('\nEpoch: %d' % epoch)
    
    total_loss = 0
    for batch_idx, (inputs1, values_original,  targets) in enumerate(trainloader): #inputs4, inputs5,
        values1 = values_original

        inputs_original = torch.cat((inputs1,inputs1,inputs1),dim=0).to(device) #,inputs4,inputs
        

        #values = torch.abs(torch.cat((values_mix2,values_mix,values1),dim=0)).to(device)
        values = torch.abs(torch.cat((values1,values1,values1),dim=0)).to(device)

        #targets = targets.to(device)
        targets = torch.cat((targets,targets,targets),dim=0).to(device) 
        S = torch.zeros_like(inputs_original,dtype=torch.float) #.to(device)
        
        ix = torch.randint(max_fea-block_size-1,(S.shape[0],)) # 20 is # of max features
        
        max_ix = torch.max(ix)
        sorted_indices_predicted = torch.zeros_like(inputs_original)
        sorted_indices_predicted[:,-1] = init_fea
        ####################################################
        input_batch_size = inputs_original.shape[0]

        
        
        inputs_all = torch.repeat_interleave(inputs_original,block_size,dim=0)
        net.eval()
        #net_gpt.eval()
        model_gpt.eval()
        with torch.no_grad():
                S = torch.zeros((input_batch_size,number_of_actions),dtype=torch.float).to(device)
                S[:,init_fea] = 1.0
                S_new = S
                inputs = mask_layer(inputs_original, S_new)
                targets_gpt = torch.zeros((S.shape[0],1,1)).to(device)
                actions = (torch.ones((S.shape[0],1,1))*init_fea).to(device)
                timesteps = torch.ones((S.shape[0],1,1)).to(device).to(torch.int64)
                count_block = 2
                for ijk in range(max_val_fea): #instead of max_fea there was 30
                    
                    count_img = 0
                    
                   
                    outputs, state_embeddings = net(inputs,fea_return=True) #,S_reshaped

                    rewards = F.softmax(outputs,dim=1).detach()
                    out_gpt = model_gpt(state_embeddings = state_embeddings, actions = actions, targets = targets_gpt, rtgs = rewards, timesteps = timesteps)
                    out_gpt = out_gpt.reshape(-1, out_gpt.size(-1))
                    out_gpt = out_gpt - S_new*1e6


                    out_gpt = out_gpt[torch.arange(count_block-2,out_gpt.shape[0],count_block-1)]
                    _, out_gpt_pred = torch.max(out_gpt, 1)

                    
                    if ijk == 0:
                        out_gpt_pred = torch.ones_like(out_gpt_pred,device=device)
                        out_gpt_pred = out_gpt_pred*second_fea #third_fea
                    if ijk == 1:
                        out_gpt_pred = torch.ones_like(out_gpt_pred,device=device)
                        out_gpt_pred = out_gpt_pred*third_fea
                    
                    sorted_indices_predicted[torch.arange(sorted_indices_predicted.shape[0]),-(ijk+2)] = out_gpt_pred.to(torch.float)
                    if actions.shape[1] < block_size:
                        actions_to_add = torch.ones((input_batch_size,1,1),device=device)
                        actions_to_add[torch.arange(input_batch_size),0,0] = out_gpt_pred.to(torch.float)
                        
                        actions = torch.cat((actions,actions_to_add),dim=1)
                        S_old = S_new
                        S_new = torch.repeat_interleave(S,count_block,dim=0)
                        #S_new[torch.arange(count_block-1,S_new.shape[0],count_block),:] = S_new[torch.arange(count_block-2,S_new.shape[0],count_block),:]
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
                    else:
                        predicted=predicted[torch.arange(count_block-3,predicted.shape[0],count_block-2)] #count_block-2::count_block-1
                   
                   
        sorted_indices_predicted = sorted_indices_predicted.to(torch.int32)
        net.train()
        model_gpt.train()
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
        
        inputs = mask_layer(inputs_all, S_new)
        indices_last = torch.arange(block_size-1, S.shape[0]*block_size, step=block_size)
        inputs_last_time_index = inputs[indices_last,:]
        outputs_kd = net(inputs_last_time_index)
        temp_val = 2
        outputs_kd = torch.nn.functional.softmax(outputs_kd/temp_val,dim=1).detach()


        outputs, state_embeddings = net(inputs,fea_return=True) #,S_reshaped
        loss = (1.0)*criterion(outputs, torch.repeat_interleave(targets,block_size,dim=0))
        #loss += (0.4)*temp_val*temp_val*criterion(outputs/temp_val, torch.repeat_interleave(outputs_kd,block_size,dim=0))
        rewards = F.softmax(outputs,dim=1).detach()
        #out_gpt = model_gpt(states = inputs, actions = actions, targets = targets_gpt, rtgs = rewards, timesteps = timesteps)                
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

def test(epoch):
    #global best_acc
    net.eval()
    #net_gpt.eval()
    model_gpt.eval()
    test_loss = 0
    correct = torch.zeros(max_val_fea) 
    total = 0
    pred_list = [[] for _ in range(max_val_fea)]
    label_list = []
    with torch.no_grad():
        for batch_idx, (inputs_original, _, targets) in enumerate(valloader):
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
                for ijk in range(max_val_fea): #instead of max_fea there was 30
                    
                    count_img = 0
                    
                   
                    outputs, state_embeddings = net(inputs,fea_return=True) #,S_reshaped

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


    metric_score = 0
    for i in range(max_val_fea):
        pred = torch.cat(pred_list[i], 0)
        y = torch.cat(label_list, 0)
        metric_score += metric(pred, y)

    print(metric_score/max_val_fea)
    return test_loss, metric_score


    

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_period = 2
    if (epoch)%val_period==0 or (epoch==(args.n_epochs-1)):
        val_loss, acc = test(epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),args.net_ckpt.replace('.ckpt', '_second_stage.ckpt')) 
            torch.save(model_gpt.state_dict(),args.GPT_ckpt.replace('.ckpt', '_second_stage.ckpt') )
        
    
    scheduler.step(epoch-1) # step cosine scheduling      
    
    if (epoch)%val_period==0 or (epoch==(args.n_epochs-1)) :
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})
    else:
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})

    
