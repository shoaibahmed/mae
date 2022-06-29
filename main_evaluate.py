# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE evaluation for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    
    parser.add_argument("--imagenet_c_path", type=str, default="/ds/images/imagenet-C/", metavar="DIR",
                        help="path to ImageNet-C")

    return parser


def main_eval(args):
    assert args.resume is not None
    
    # Initialize the distributed environment
    misc.init_distributed_mode(args)
    
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(args.device)

    cudnn.benchmark = True

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    # Create the model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    ).to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    # Setting them to none is fine as far as they are not in the state_dict
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None)

    # linear probe: weak augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val_folders'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    if misc.is_main_process():
        imagenet_c_stats_file = open(f'{args.model}_imagenet_c_stats.txt', 'w', buffering=1)
    
    train_stats = evaluate(data_loader_train, model, device)
    print(f"Accuracy of the network on the {len(dataset_train)} training images: {train_stats['acc1']:.1f}%")
    if misc.is_main_process():
        stats = dict(set="Train")
        stats.update(train_stats)
        print(json.dumps(stats))
        print(json.dumps(stats), file=imagenet_c_stats_file)
    
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    if misc.is_main_process():
        stats = dict(set="Validation")
        stats.update(test_stats)
        print(json.dumps(stats))
        print(json.dumps(stats), file=imagenet_c_stats_file)
    
    # Evaluate on ImageNet-C
    noise_classes = [os.path.join(args.imagenet_c_path, x) for x in os.listdir(args.imagenet_c_path)]
    noise_dirs = []
    noise_class_names = []
    for cls in noise_classes:
        if not os.path.isdir(cls):
            continue
        current_dirs = [os.path.join(cls, dir) for dir in os.listdir(cls)]
        assert all([os.path.exists(x) for x in current_dirs])
        noise_dirs += current_dirs
        noise_class_names += [dir for dir in os.listdir(cls)]
    
    if misc.is_main_process():
        print("Final noise dirs:", noise_dirs)
        print("Noise class names:", noise_class_names)
    
    for dir, cls_name in zip(noise_dirs, noise_class_names):
        cls_name = cls_name.replace("_", " ").title()
        if misc.is_main_process():
            print("Loading dir:", dir)
            print("Noise class:", cls_name)
        
        for severity in ["1", "2", "3", "4", "5"]:
            final_dir = os.path.join(dir, severity)
            assert os.path.exists(final_dir)
            if misc.is_main_process():
                print("Severity:", severity)
                print("Final directory loaded:", final_dir)
            
            dataset_im_c = datasets.ImageFolder(final_dir, transform=transform_val)
            im_c_loader = torch.utils.data.DataLoader(
                dataset_im_c, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
            
            val_stats = evaluate(im_c_loader, model, device)
            print(f"Accuracy of the network on the {len(dataset_im_c)} images from {cls_name} with severity {severity}: {val_stats['acc1']:.1f}%")
            
            if misc.is_main_process():
                stats = dict(noise_class=cls_name, severity=severity, noise_dir=final_dir)
                stats.update(val_stats)  # Concatenate test stats
                print(json.dumps(stats))
                print(json.dumps(stats), file=imagenet_c_stats_file)
    
    if misc.is_main_process():
        imagenet_c_stats_file.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_eval(args)
