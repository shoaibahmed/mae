#!/bin/bash

echo "Pretraining stagewise MAE (limited schedule)"
job=simultaneous_stagewise_mae_base_pretrain_limited
srun -p A100 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_pretrain_stagewise.py \
        --model stagewise_mae_vit_base_patch16 \
        --data_path /ds/images/imagenet/ \
        --batch_size 64 \
        --accum_iter 8 \
        --mask_ratio "0.2;0.4;0.6;0.75" \
        --epochs 100 \
        --warmup_epochs 40 \
        --norm_pix_loss \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir ./outputs/${job} --log_dir ./outputs/${job} \
        --num_workers 8 --pin_mem > ./logs/${job}.log 2>&1 &

echo "Pretraining multihead MAE (limited schedule)"
job=simultaneous_multihead_mae_base_pretrain_limited
srun -p A100 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_pretrain_multihead.py \
        --model multihead_mae_vit_base_patch16 \
        --data_path /ds/images/imagenet/ \
        --batch_size 64 \
        --accum_iter 8 \
        --mask_ratio "0.2;0.4;0.6;0.75" \
        --epochs 100 \
        --warmup_epochs 40 \
        --norm_pix_loss \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir ./outputs/${job} --log_dir ./outputs/${job} \
        --num_workers 8 --pin_mem > ./logs/${job}.log 2>&1 &

echo "Pretraining multihead MAE with same masking ratio (limited schedule)"
job=simultaneous_multihead_mae_base_pretrain_limited_constant_mask_ratio
srun -p A100 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_pretrain_multihead.py \
        --model multihead_mae_vit_base_patch16 \
        --data_path /ds/images/imagenet/ \
        --batch_size 64 \
        --accum_iter 8 \
        --mask_ratio "0.75;0.75;0.75;0.75" \
        --epochs 100 \
        --warmup_epochs 40 \
        --norm_pix_loss \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir ./outputs/${job} --log_dir ./outputs/${job} \
        --num_workers 8 --pin_mem > ./logs/${job}.log 2>&1 &

echo "Pretraining MAE (limited schedule)"
job=mae_base_pretrain_limited
srun -p A100 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_pretrain.py \
        --model mae_vit_base_patch16 \
        --data_path /ds/images/imagenet/ \
        --batch_size 64 \
        --accum_iter 8 \
        --mask_ratio 0.75 \
        --epochs 100 \
        --warmup_epochs 40 \
        --norm_pix_loss \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir ./outputs/${job} --log_dir ./outputs/${job} \
        --num_workers 8 --pin_mem > ./logs/${job}.log 2>&1 &

echo "Pretraining stagewise MAE with same masking ratio (limited schedule)"
job=simultaneous_stagewise_mae_base_pretrain_limited_constant_mask_ratio
srun -p A100 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_pretrain_stagewise.py \
        --model stagewise_mae_vit_base_patch16 \
        --data_path /ds/images/imagenet/ \
        --batch_size 64 \
        --accum_iter 8 \
        --mask_ratio "0.75;0.75;0.75;0.75" \
        --epochs 100 \
        --warmup_epochs 40 \
        --norm_pix_loss \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir ./outputs/${job} --log_dir ./outputs/${job} \
        --num_workers 8 --pin_mem > ./logs/${job}.log 2>&1 &
