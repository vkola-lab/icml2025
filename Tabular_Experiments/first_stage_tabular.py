# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import time
import argparse
from fastshap.utils import MaskLayer
from copy import deepcopy
from tabular_dataset import get_dataset, data_split
import os
from gpt_model import GPT, GPTConfig
print(os.getenv("CUDA_VISIBLE_DEVICES"))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from torch.utils.data import Dataset
from torchmetrics import  Accuracy, AUROC
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
parser.add_argument('--n_epochs', type=int, default='200')

parser.add_argument('--init_fea',  type=int)
parser.add_argument('--second_fea',  type=int)
parser.add_argument('--third_fea',  type=int)

parser.add_argument('--wandb_name') 
parser.add_argument('--wandb_project') 
parser.add_argument('--dataset_path')
args = parser.parse_args()



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
class TabularSHAP(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset        
        #self.shap_vals = shap_vals
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, x_shap, y = self.dataset[idx]
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
            device,
            mask_size,
            nepochs,
            loss_fn,
            val_metric_fn=None,
            val_metric_mode='max',
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
        if val_metric_fn is None:
            val_metric_fn = loss_fn
            val_metric_mode = 'min'

        
        # Set up optimizer and lr scheduler.
        model = self.model
        mask_layer = self.mask_layer
        #device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_metric_mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose)
        
        best_model = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 8
            
        for epoch in range(nepochs):
            # Switch model to training mode.
            model.train()

            for x, _, y in train_loader:
                # Move to device.
                x = x.to(device)
                y = y.to(device)
                
                # Generate missingness.
                m = generate_uniform_mask(len(x), mask_size).to(device)
                #m = imp.resize(m)
                # Calculate loss.
                x_masked = mask_layer(x, m)
                #print(x_masked.shape)
                pred = model(x_masked)
                loss = loss_fn(pred, y)

                # Take gradient step.
                loss.backward()
                opt.step()
                model.zero_grad()
                
            # Calculate validation loss.
            model.eval()
            total_correct_val = 0
            total_samples_val = 0
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for x, _, y in val_loader:
                    # Move to device.
                    x = x.to(device)
                    
                    # Generate missingness.
                    # TODO this should be precomputed and shared across epochs
                    m = generate_uniform_mask(len(x), mask_size).to(device)
                    #m = imp.resize(m)
                    # Calculate prediction.
                    x_masked = mask_layer(x, m)
                    pred = model(x_masked)
                    pred_list.append(pred.cpu())
                    label_list.append(y.cpu())
                    
                    _, predicted = torch.max(pred.data, 1)
                    predicted = predicted.to(device)
                    y=y.to(device)
                    total_samples_val += y.size(0)
                    total_correct_val += (predicted == y).sum().item()

                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_metric = val_metric_fn(pred, y).item()
                
            
            # Print progress.
            if verbose:
                print(f'{"-"*8}Epoch {epoch+1}{"-"*8}')
                print(f'Val Metric = {val_metric:.4f}\n')
                
            # Update scheduler.
            scheduler.step(val_metric)

            # Check if best model.
            if val_metric == scheduler.best:
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

# Initialize wandb 
usewandb = ~args.nowandb
if usewandb:
    import wandb
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

pre_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8) # generator=g) #bs #32





init_fea=args.init_fea
second_fea=args.second_fea
third_fea=args.third_fea

train_dataset = TabularSHAP(train_dataset)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8) #, generator=g) #bs #32
valloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8) #, generator=g)

net = MLP_tabular(d_in=2*d_in,d_out=d_out, n_hidden=args.n_hidden, n_embd=args.n_embd) 
net = net.to(device)

print(net)
# Set up mini-GPT model

number_of_actions = d_in #vocab_size
block_size = args.n_blocks
nhead = args.n_head


max_fea = args.max_fea 
max_val_fea = args.max_val_fea

mask_layer = MaskLayer(append=True)
pretrain = MaskingPretrainer(net, mask_layer).to(device)


# print('beginning pre-training...')
pretrain.fit(
    pre_trainloader,
    valloader,
    lr=1e-3,
    device=device,
    mask_size=number_of_actions,
    patience=3,
    nepochs=250,
    loss_fn=nn.CrossEntropyLoss(),
    val_metric_fn=metric,
    verbose=True)
print('done pretraining')


mconf = GPTConfig(number_of_actions,  args.n_blocks,
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, model_type='reward_conditioned', max_timestep=number_of_actions)
model_gpt = GPT(mconf,d_out) #conf,state_emb_net,n_class
model_gpt = model_gpt.to(device)
# Loss is CE
criterion = nn.CrossEntropyLoss()
if args.opt == "adam":
    optimizer = optim.Adam(set(list(net.parameters()) +list(model_gpt.parameters())), lr=1e-3, weight_decay=1e-3) # lr=1e-3 + list(net_gpt.parameters()) #
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  

# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
use_amp = not args.noamp
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
soft_func = nn.LogSoftmax(dim=1)
max_float16 = 65504 #3.4028235e+38 #np.finfo(np.float32).max


# The prediction network is trained in a classical way
# The policy network (mini-GPT) is trained as the next token predcition, i.e. next action prediction
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    #net_gpt.train()
    model_gpt.train()
    train_loss = 0
    correct = 0
    total = 0
    total_loss = 0
    for batch_idx, (inputs1, values_original,targets) in enumerate(trainloader): #inputs4, inputs5,
        values1 = values_original #[np.arange(values_original.shape[0]),:,np.array((targets))].reshape((values_original.shape[0],number_of_actions))
        values = torch.cat((values1,values1,values1),dim=0)

        sorted_indices = torch.argsort(torch.abs(values), dim=1).to(device)

        
        inputs1 = inputs1.to(device)
        inputs = torch.cat((inputs1,inputs1,inputs1),dim=0)
        targets = torch.cat((targets,targets,targets),dim=0).to(device) 
        
        
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
        inputs = torch.repeat_interleave(inputs,block_size,dim=0)
        inputs = mask_layer(inputs, S_new)
        outputs, state_embeddings = net(inputs,fea_return=True) #,S_reshaped
        loss = (1.0)*criterion(outputs, torch.repeat_interleave(targets,block_size,dim=0))

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

##### Validation
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
        for batch_idx, (inputs_original,_, targets) in enumerate(valloader):
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
                   
                    correct[ijk] += predicted.eq(targets).sum().item()
    metric_score = 0
    for i in range(max_val_fea):
        pred = torch.cat(pred_list[i], 0)
        y = torch.cat(label_list, 0)
        metric_score += metric(pred, y)
    correct = torch.mean(correct).item()
    print(metric_score/max_val_fea)
    return test_loss, metric_score


best_acc = 0  # best test accuracy
start_epoch = 0     


for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    if (epoch+1)%5==0:
        val_loss, acc = test(epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),args.net_ckpt )
            torch.save(model_gpt.state_dict(),args.GPT_ckpt )
           
    scheduler.step(epoch-1) 
    
    # Log training..
    if (epoch+1)%5==0:
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})
    else:
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})

    
