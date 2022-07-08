#!/bin/bash

python3 plot_bar_chart.py \
    "./vit_base_patch16_imagenet_c_stats.txt" \
    "./vit_huge_patch14_imagenet_c_stats.txt" \
    "ViT-Base;ViT-Huge" \
    "imagenet_c_mae.png"
