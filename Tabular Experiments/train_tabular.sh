#!/bin/bash -l

conda activate /projectnb/vkolagrp/osman

dataset="cps"
if [ "$dataset" = "metabric" ]; then
    init_fea=156
    second_fea=25
    third_fea=359
    n_hidden=1024
    n_embd=512
    max_fea=35
    max_val_fea=20
elif [ "$dataset" = "ctgs" ]; then
    init_fea=0
    second_fea=18
    third_fea=20
    n_hidden=512
    n_embd=128
    max_fea=22
    max_val_fea=20
elif [ "$dataset" = "spam" ]; then
    init_fea=52
    second_fea=6
    third_fea=15
    n_hidden=512
    n_embd=128
    max_fea=35
    max_val_fea=20
elif [ "$dataset" = "ckd" ]; then
    init_fea=19
    second_fea=21
    third_fea=42
    n_hidden=512
    n_embd=128
    max_fea=35
    max_val_fea=30
elif [ "$dataset" = "cps" ]; then
    init_fea=3
    second_fea=0
    third_fea=6
    n_hidden=128
    n_embd=64
    max_fea=7
    max_val_fea=7
else
    echo "Invalid dataset selected"
    exit 1
fi

wandb_project=${dataset} # CIFAR100
wandb_name="v1_first_stage" 

n_layer=3
n_head=4
n_blocks=4
lr=1e-3
n_epochs=200
#lr=5e-4

net_ckpt="/projectnb/vkolagrp/projects/active_feature_acquisition/codes/ckpts/net_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nhidden${n_hidden}_nembd${n_embd}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"
GPT_ckpt="/projectnb/vkolagrp/projects/active_feature_acquisition/codes/ckpts/GPT_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nhidden${n_hidden}_nembd${n_embd}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"
GPT_fea_ckpt="/projectnb/vkolagrp/projects/active_feature_acquisition/codes/ckpts/GPT_fea_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nhidden${n_hidden}_nembd${n_embd}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"



python first_stage_tabular.py  --n_layer ${n_layer} --n_head ${n_head} --n_blocks ${n_blocks} --lr ${lr} \
                                  --n_hidden ${n_hidden} --n_embd ${n_embd} \
                                  --n_epochs ${n_epochs} \
                                  --shared_backbone \
                                  --wandb_project ${wandb_project} --wandb_name ${wandb_name} \
                                  --max_fea ${max_fea} --max_val_fea ${max_val_fea} \
                                  --init_fea ${init_fea} --second_fea ${second_fea} --third_fea ${third_fea}  \
                                  --dataset ${dataset} \
                                  --net_ckpt ${net_ckpt} --GPT_ckpt ${GPT_ckpt} --GPT_fea_ckpt ${GPT_fea_ckpt}

wandb_name="v1_second_stage" 
n_epochs_second=16

python second_stage_tabular.py  --n_layer ${n_layer} --n_head ${n_head} --n_blocks ${n_blocks} --lr ${lr} \
                                  --n_hidden ${n_hidden} --n_embd ${n_embd} \
                                  --n_epochs ${n_epochs_second} \
                                  --shared_backbone \
                                  --wandb_project ${wandb_project} --wandb_name ${wandb_name} \
                                  --max_fea ${max_fea} --max_val_fea ${max_val_fea} \
                                  --init_fea ${init_fea} --second_fea ${second_fea} --third_fea ${third_fea}  \
                                  --dataset ${dataset} \
                                  --net_ckpt ${net_ckpt} --GPT_ckpt ${GPT_ckpt} --GPT_fea_ckpt ${GPT_fea_ckpt}
