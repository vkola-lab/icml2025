#!/bin/bash -l

conda activate /projectnb/vkolagrp/osman


dataset="imagenette"  # Default dataset

if [ "$dataset" = "cifar100" ]; then
    max_fea=20
    image_size=32
    patch_size=4
    num_classes=100
    lr=5e-4
    pretrain_lr=5e-4
    second_stage_lr=5e-4
    surr_ckpt="ckpts/AFA-XAI/cifar100_surrogate_size4.pt"
    explainer_ckpt="ckpts/AFA-XAI/cifar100_explainer_size4.pt"
    init_fea=35
    second_fea=36
    third_fea=27
elif [ "$dataset" = "cifar10" ]; then
    max_fea=32
    image_size=32
    patch_size=4
    num_classes=10
    lr=1e-3
    pretrain_lr=1e-3
    second_stage_lr=1e-3
    surr_ckpt="ckpts/AFA-XAI/cifar10_surrogate_size4.pt" #cifar10_surrogate_size4
    explainer_ckpt="ckpts/AFA-XAI/cifar10_explainer_size4.pt"
    init_fea=27
    second_fea=35
    third_fea=28
elif [ "$dataset" = "bloodmnist" ]; then
    max_fea=20
    image_size=28
    patch_size=2
    num_classes=8
    lr=1e-3
    pretrain_lr=1e-3
    second_stage_lr=1e-3
    surr_ckpt="ckpts/AFA-XAI/bloodmnist_surrogate_size2.pt"
    explainer_ckpt="ckpts/AFA-XAI/bloodmnist_explainer_size2.pt"
    init_fea=90
    second_fea=132
    third_fea=118
elif [ "$dataset" = "imagenette" ]; then  
    export FASTAI_HOME=./datasets/
    max_fea=50
    image_size=224
    patch_size=16
    num_classes=10    
    lr=5e-4
    pretrain_lr=5e-4
    second_stage_lr=1e-5
    surr_ckpt="ckpts/AFA-XAI/imagenette_surrogate_size16.pt"
    explainer_ckpt="ckpts/AFA-XAI/imagenette_explainer_size16.pt"
    init_fea=103
    second_fea=107
    third_fea=118    
else
    echo "Invalid dataset selected"
    exit 1
fi




wandb_project=${dataset} # CIFAR100
wandb_name="v1_first_stage" 
max_val_fea=20
n_layer=3
n_head=4
n_blocks=4

n_epochs=200
#lr=5e-4

net_ckpt="ckpts/net_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"
GPT_ckpt="ckpts/GPT_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"
GPT_fea_ckpt="ckpts/GPT_fea_three_fea_nlayer${n_layer}_nhead${n_head}_nblocks${n_blocks}_nepochs${n_epochs}_maxfea${max_fea}_${dataset}_lr${lr}.ckpt"

# Print the checkpoint path (optional, for debugging)
echo "Checkpoint path: $net_ckpt"
# echo $wandb_name

python first_stage_image_datasets.py  --n_layer ${n_layer} --n_head ${n_head} --n_blocks ${n_blocks} --lr ${lr} --pretrain_lr ${pretrain_lr} \
                                  --surr_ckpt ${surr_ckpt} --explainer_ckpt ${explainer_ckpt} \
                                  --n_epochs ${n_epochs} \
                                  --shared_backbone \
                                  --wandb_project ${wandb_project} --wandb_name ${wandb_name} \
                                  --max_fea ${max_fea} --num_classes ${num_classes} \
                                  --init_fea ${init_fea} --second_fea ${second_fea} --third_fea ${third_fea}  \
                                  --dataset ${dataset} \
                                  --image_size ${image_size} --patch_size ${patch_size} \
                                  --net_ckpt ${net_ckpt} --GPT_ckpt ${GPT_ckpt} --GPT_fea_ckpt ${GPT_fea_ckpt} 

n_epochs_second=16
wandb_name="v1_second_stage" 
python second_stage_image_datasets.py  --n_layer ${n_layer} --n_head ${n_head} --n_blocks ${n_blocks} --lr ${second_stage_lr} \
                                  --surr_ckpt ${surr_ckpt} --explainer_ckpt ${explainer_ckpt} \
                                  --n_epochs ${n_epochs_second} \
                                  --shared_backbone \
                                  --wandb_project ${wandb_project} --wandb_name ${wandb_name} \
                                  --max_fea ${max_fea} --num_classes ${num_classes} \
                                  --init_fea ${init_fea} --second_fea ${second_fea} --third_fea ${third_fea}  \
                                  --dataset ${dataset} \
                                  --image_size ${image_size} --patch_size ${patch_size} \
                                  --net_ckpt ${net_ckpt} --GPT_ckpt ${GPT_ckpt} --GPT_fea_ckpt ${GPT_fea_ckpt}



