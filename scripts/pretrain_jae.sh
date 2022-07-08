#!/bin/bash

# python submitit_pretrain.py \
#     --job_dir ${JOB_DIR} \
#     --nodes 8 \
#     --use_volta32 \
#     --batch_size 64 \
#     --model mae_vit_large_patch16 \
#     --norm_pix_loss \
#     --mask_ratio 0.75 \
#     --epochs 800 \
#     --warmup_epochs 40 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path ${IMAGENET_DIR}

# Batch size: 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096

echo "Pretraining JAE"
job=jae_base_pretrain
srun -p RTXA6000 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time=24-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_pretrain_jigsaw.py \
        --model jae_vit_base_patch16 \
        --data_path /ds/images/imagenet/ \
        --batch_size 256 \
        --accum_iter 2 \
        --epochs 800 \
        --warmup_epochs 40 \
        --norm_pix_loss \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir ./outputs/jae_base_pretrain --log_dir ./outputs/jae_base_pretrain \
        --num_workers 8 --pin_mem > ./logs/${job}.log 2>&1 &
