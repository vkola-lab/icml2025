#!/bin/bash -l

# run this script from adrd_tool/

conda activate /projectnb/vkolagrp/osman


dataset="metabric"
init_fea=156
second_fea=25
third_fea=359







n_hidden=1024
n_embd=512


wandb_project=${dataset} # CIFAR100
wandb_name="v1" 
max_fea=35
max_val_fea=20
n_layer=3
n_head=4
n_blocks=4
lr=1e-3
n_epochs=200
#lr=5e-4

net_ckpt="/projectnb/vkolagrp/projects/active_feature_acquisition/codes/ckpts/net_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nhidden${n_hidden}_nembd${n_embd}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"
GPT_ckpt="/projectnb/vkolagrp/projects/active_feature_acquisition/codes/ckpts/GPT_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nhidden${n_hidden}_nembd${n_embd}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"
GPT_fea_ckpt="/projectnb/vkolagrp/projects/active_feature_acquisition/codes/ckpts/GPT_fea_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nhidden${n_hidden}_nembd${n_embd}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"

# Print the checkpoint path (optional, for debugging)
echo "Checkpoint path: $net_ckpt"
# echo $wandb_name

python first_stage_tabular.py  --n_layer ${n_layer} --n_head ${n_head} --n_blocks ${n_blocks} --lr ${lr} \
                                  --n_hidden ${n_hidden} --n_embd ${n_embd} \
                                  --n_epochs ${n_epochs} \
                                  --shared_backbone \
                                  --wandb_project ${wandb_project} --wandb_name ${wandb_name} \
                                  --max_fea ${max_fea} \
                                  --init_fea ${init_fea} --second_fea ${second_fea} --third_fea ${third_fea}  \
                                  --dataset ${dataset} \
                                  --net_ckpt ${net_ckpt} --GPT_ckpt ${GPT_ckpt} --GPT_fea_ckpt ${GPT_fea_ckpt}





