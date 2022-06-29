#!/bin/bash

python main_evaluate.py --model vit_huge_patch14 \
    --data_path /mnt/sas/Datasets/ilsvrc12/ \
    --imagenet_c_path /mnt/sas/Datasets/imagenet_c_tar/complete/ \
    --resume ./checkpoints/mae_finetuned_vit_huge.pth \
    --num_workers 8 --pin_mem
