#!/bin/bash

srun -p RTX3090 -K -N1 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
    --kill-on-bad-exit --job-name "mae_eval" --nice=0 --time=3-00:00:00 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python main_evaluate.py --model vit_huge_patch14 \
        --data_path /ds/images/imagenet/ \
        --imagenet_c_path /ds/images/imagenet-C/ \
        --resume ./checkpoints/mae_finetuned_vit_huge.pth \
        --num_workers 8 --pin_mem --batch_size 128 --dist_eval > ./logs/mae_eval_vit_h.log 2>&1 &
