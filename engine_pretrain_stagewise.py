# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_list, _, _ = model(samples, mask_ratio_list=args.mask_ratio)

        loss_value_list = [loss.item() for loss in loss_list]

        if not all([math.isfinite(x) for x in loss_value_list]):
            print("Loss is {}, stopping training".format(loss_value_list))
            sys.exit(1)

        for idx, loss in enumerate(loss_list):
            loss /= accum_iter
            update_grad = ((data_iter_step + 1) % accum_iter == 0) and (idx == len(loss_list) - 1)
            print(f"Block idx: {idx+1} / Loss: {float(loss):.4f} / Update grad: {update_grad}")
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=update_grad)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{f"loss_block_{i}": loss_value_list[i] for i in range(len(loss_value_list))})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce_list = [misc.all_reduce_mean(x) for x in loss_value_list]
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            for block in range(len(loss_value_reduce_list)):
                log_writer.add_scalar(f'train_loss_block_{block}', loss_value_reduce_list[block], epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}