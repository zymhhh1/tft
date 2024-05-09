# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import os
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
# from apex.optimizers import FusedAdam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp

import numpy as np
import pandas as pd

import dllogger

from modeling import TemporalFusionTransformer
from configuration import CONFIGS
from data_utils import load_dataset
from log_helper import setup_logger
from inference import predict
from utils import PerformanceMeter, print_once
# import gpu_affinity
from ema import ModelEma

from sklearn.metrics import classification_report 

def main(args):
    ### INIT DISTRIBUTED
    args.distributed_world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    torch.backends.cudnn.benchmark = True

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    setup_logger(args)

    config = CONFIGS[args.dataset]()
    if args.overwrite_config:
        config.__dict__.update(json.loads(args.overwrite_config))

    dllogger.log(step='HPARAMS', data={**vars(args), **vars(config)}, verbosity=1)

    train_loader, valid_loader, test_loader = load_dataset(args, config)

    model = TemporalFusionTransformer(config).cuda()
    if args.ema_decay:
        model_ema = ModelEma(model, decay=args.ema_decay)

    # Run dummy iteration to initialize lazy modules
    dummy_batch = next(iter(train_loader))
    dummy_batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in dummy_batch.items()}
    model(dummy_batch)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print_once('Model params: {}'.format(sum(p.numel() for p in model.parameters())))
    global_step = 0
    perf_meter = PerformanceMeter(benchmark_mode=not args.disable_benchmark)
    if args.use_amp:
        scaler = amp.GradScaler(init_scale=32768.0)

    for epoch in range(args.epochs):
        real_list = []
        pred_list = []
        loss_list = []
        start = time.time()
        dllogger.log(step=global_step, data={'epoch': epoch}, verbosity=1)

        model.train() 
        ii = 0
        for local_step, batch in enumerate(train_loader):
            if ii % 1000 == 0:
                print(ii)
            ii += 1
            perf_meter.reset_current_lap()
            batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
            with torch.jit.fuser("fuser2"), amp.autocast(enabled=args.use_amp):
                predictions = model(batch)
                targets = batch['target'][:,config.encoder_length:,:]
                #print(f"predictions.shape:{predictions.shape}, targets.shape:{targets.shape}")
                #print(f"predictions.shape:{predictions.reshape(-1, 5).shape}, targets.shape:{targets.reshape(-1).shape}")
                p_losses = criterion(predictions.reshape(-1, 5), targets.reshape(-1))
                pred_list.extend(np.argmax(predictions.detach().cpu().numpy().reshape(-1, 5), axis=1).reshape(-1).tolist())
                real_list.extend(targets.cpu().numpy().reshape(-1).tolist())
                loss = p_losses.sum()
                loss_list.append(p_losses.sum().item())
            if global_step == 0 and args.ema_decay:
                model_ema(batch)
            if args.use_amp:
                scaler.scale(loss).backward()

            else:
                loss.backward()
            if not args.grad_accumulation or (global_step+1) % args.grad_accumulation == 0:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                if args.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if args.ema_decay:
                    model_ema.update(model)

            torch.cuda.synchronize()
            '''ips = perf_meter.update(args.batch_size * args.distributed_world_size,
                    exclude_from_total=local_step in [0, 1, 2, len(train_loader)-1])

            log_dict = {'P10':p_losses[0].item(), 'P50':p_losses[1].item(), 'P90':p_losses[2].item(), 'loss': loss.item(), 'items/s':ips}
            dllogger.log(step=global_step, data=log_dict, verbosity=1)'''
            global_step += 1
        print("train: loss:{}".format(sum(loss_list) / len(loss_list)))
        print(classification_report(real_list, pred_list))
        validate(args, config, model_ema if args.ema_decay else model, criterion, valid_loader, global_step)

        if validate.early_stop_c >= args.early_stopping:
            print_once('Early stopping')
            break

    ### TEST PHASE ###
    state_dict = torch.load(os.path.join(args.results, 'checkpoint.pt'), map_location='cpu')
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict['model'])
    model.cuda().eval()

    #tgt_scalers = pickle.load(open(os.path.join(args.data_path, 'tgt_scalers.bin'), 'rb'))
    #cat_encodings = pickle.load(open(os.path.join(args.data_path,'cat_encodings.bin'), 'rb'))

    unscaled_predictions, unscaled_targets, _, _ = predict(args, config, model, test_loader)
    unscaled_predictions = torch.from_numpy(unscaled_predictions).contiguous()
    unscaled_predictions_max = np.argmax(unscaled_predictions, axis=-1).reshape(-1)
    unscaled_targets = torch.from_numpy(unscaled_targets).contiguous().reshape(-1)
    #print(unscaled_predictions)
    #print(unscaled_targets)
    pred_probability = [','.join([str(i.numpy()) for i in l]) for l in
        F.softmax(unscaled_predictions.reshape(-1, unscaled_predictions.shape[-1]), dim=-1)]
    pd.DataFrame({
        'pred': unscaled_predictions_max,
        'pred_probability': pred_probability,
        'real': unscaled_targets,
    }).to_csv('results/result.csv')

    '''losses = nn.CrossEntropyLoss(unscaled_predictions, unscaled_targets)
    normalizer = unscaled_targets.abs().mean()
    quantiles = 2 * losses / normalizer

    quantiles = {'test_p10': quantiles[0].item(), 'test_p50': quantiles[1].item(), 'test_p90': quantiles[2].item(), 'sum':sum(quantiles).item()}
    finish_log = {**quantiles, 'average_ips':perf_meter.avg, 'convergence_step':validate.conv_step}
    dllogger.log(step=(), data=finish_log, verbosity=1)'''

def validate(args, config, model, criterion, dataloader, global_step):
    if not hasattr(validate, 'best_valid_loss'):
        validate.best_valid_loss = float('inf')
    if not hasattr(validate, 'early_stop_c'):
        validate.early_stop_c = 0
    model.eval()

    losses = []
    real_list = []
    pred_list = []
    torch.cuda.synchronize()
    validation_start = time.time()
    for batch in dataloader:
        with torch.jit.fuser("fuser2"), amp.autocast(enabled=args.use_amp), torch.no_grad():
            batch = {key: tensor.cuda() if tensor.numel() else None for key, tensor in batch.items()}
            predictions = model(batch)
            targets = batch['target'][:,config.encoder_length:,:]
            p_losses = criterion(predictions.reshape(-1, 5), targets.reshape(-1))
            bs = next(t for t in batch.values() if t is not None).shape[0]
            losses.append((p_losses, bs))
            pred_list.extend(np.argmax(predictions.detach().cpu().numpy().reshape(-1, 5), axis=1).reshape(-1).tolist())
            real_list.extend(targets.cpu().numpy().reshape(-1).tolist())

    torch.cuda.synchronize()
    validation_end = time.time()

    p_losses = sum([l[0]*l[1] for l in losses])/sum([l[1] for l in losses]) #takes into accunt that the last batch is not full

    ips = len(dataloader.dataset) / (validation_end - validation_start)

    #log_dict = {'P10':p_losses[0].item(), 'P50':p_losses[1].item(), 'P90':p_losses[2].item(), 'loss': p_losses.sum().item(), 'items/s':ips}

    if p_losses.sum().item() < validate.best_valid_loss:
        validate.best_valid_loss = p_losses.sum().item()
        validate.early_stop_c = 0
        validate.conv_step = global_step
        if not dist.is_initialized() or dist.get_rank() == 0:
            state_dict = model.module.state_dict() if isinstance(model, (DDP, ModelEma)) else model.state_dict()
            ckpt = {'args':args, 'config':config, 'model':state_dict}
            torch.save(ckpt, os.path.join(args.results, 'checkpoint.pt'))
    else:
        validate.early_stop_c += 1
        
    #log_dict = {'val_'+k:v for k,v in log_dict.items()}
    #dllogger.log(step=global_step, data=p_losses.sum().item(), verbosity=1)
    print("validate: loss:{}".format(p_losses.sum().item()))
    print(classification_report(real_list, pred_list))


if __name__ == '__main__':
    import os
    os.environ['PYTORCH_NVFUSER_DISABLE_FALLBACK'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=CONFIGS.keys(),
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Default number of training epochs')
    parser.add_argument('--sample_data', type=lambda x: int(float(x)), nargs=2, default=[-1, -1],
                        help="""Subsample the dataset. Specify number of training and valid examples.
                        Values can be provided in scientific notation. Floats will be truncated.""")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--grad_accumulation', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=1000,
                        help='Stop training if validation loss does not improve for more than this number of epochs.')
    parser.add_argument('--results', type=str, default='/results',
                        help='Directory in which results are stored')
    parser.add_argument('--log_file', type=str, default='dllogger.json',
                        help='Name of dllogger output file')
    parser.add_argument('--overwrite_config', type=str, default='',
                       help='JSON string used to overload config')
    parser.add_argument('--affinity', type=str,
                         default='disabled',
                         choices=['socket', 'single', 'single_unique',
                                  'socket_unique_interleaved',
                                  'socket_unique_continuous',
                                  'disabled'],
                         help='type of CPU affinity')
    parser.add_argument("--ema_decay", type=float, default=0.0, help='Use exponential moving average')
    parser.add_argument("--disable_benchmark", action='store_true', help='Disable benchmarking mode')


    ARGS = parser.parse_args()
    main(ARGS)
