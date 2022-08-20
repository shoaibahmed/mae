#!/bin/bash

# python main_finetune.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path
# Total batch size should be 1024

job=vit_base_jae_ft
srun -p RTXA6000 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time=20-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_finetune.py \
        --model vit_base_patch16 \
        --data_path /ds/images/imagenet/ \
        --finetune outputs/jae_base_pretrain/checkpoint-20.pth \
        --batch_size 128 \
        --epochs 100 \
        --blr 5e-4 --layer_decay 0.65 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
        --output_dir ./outputs/jae_base_20e_ft/ --log_dir ./outputs/jae_base_20e_ft/ \
        --num_workers 8 --pin_mem --dist_eval > ./logs/${job}.log 2>&1 &