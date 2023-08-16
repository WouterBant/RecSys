import collections
import os
import random
from pathlib import Path
import logging
import shutil
import time
import mmcv
from mmcv.runner import get_dist_info
import pickle

from packaging import version
import pandas as pd

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from param import parse_args
from utils import LossMeter
from trainer_base import TrainerBase
from pretrain_data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict, all_gather

from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast

from torch.utils.data import DataLoader, Dataset, Sampler
from pretrain_data import get_loader



_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast



def create_config(args):
    from transformers import T5Config, BartConfig
    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    config = config_class.from_pretrained(args.backbone)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from tokenization import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.backbone

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )

    print(tokenizer_class, tokenizer_name)
    return tokenizer


def create_model(model_class, config=None):
    print(f'Building Model at GPU {args.gpu}')

    model_name = args.backbone

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model

def evaluate(test_loader):
    
    from pretrain_model import P5Pretraining
    config = create_config(args)
    if args.tokenizer is None:
        args.tokenizer = args.backbone

    tokenizer = create_tokenizer(args)
    #model_kwargs = {}
    model_class = P5Pretraining
    model = create_model(model_class, config)

    if 'p5' in args.tokenizer:
        model.resize_token_embeddings(tokenizer.vocab_size)

    model.tokenizer = tokenizer

    # Load Checkpoint
    from utils import load_state_dict, LossMeter, set_global_logging_level
    from pprint import pprint

    def load_checkpoint(ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        results = model.load_state_dict(state_dict, strict=False)

        print('Model loaded from ', ckpt_path)
        print(results)

    ckpt_path = args.load
    load_checkpoint(ckpt_path)

    from MIND_templates import all_tasks as task_template
    # GPU Options
    print(f'Model Launching at GPU {args.gpu}')


    model = model.to(args.gpu)
    if args.multiGPU:
        if args.distributed:
            model = DDP(model, device_ids=[args.gpu],
                             find_unused_parameters=True
                             )

    user_each = []
    impress_each = []
    item_each = []
    truth_each = []
    pred_each = []
    prob_each = []

    rank, world_size = get_dist_info()
    print('output list length', len(test_loader) * world_size  * args.val_batch_size)
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_loader))

    for step_i, batch in enumerate(test_loader):
        with torch.no_grad():
            if args.distributed:
                # from each rank
                user_id, impress_id, item_id, target_text, prob_yes, generated_sents = model.module.generate_step(batch)

        user_each.extend(user_id)
        item_each.extend(item_id)
        impress_each.extend(impress_id)
        truth_each.extend(target_text)
        prob_each.extend(prob_yes)
        pred_each.extend(generated_sents)

        if rank == 0 and step_i % 100 == 0:
            prog_bar.update(100)

    # collect results from all ranks
    user_res = collect_results_gpu(user_each, len(test_loader) * world_size * args.val_batch_size)
    item_res = collect_results_gpu(item_each, len(test_loader) * world_size * args.val_batch_size)
    impress_res = collect_results_gpu(impress_each, len(test_loader) * world_size * args.val_batch_size)
    truth_res = collect_results_gpu(truth_each, len(test_loader) * world_size  * args.val_batch_size)
    prob_res = collect_results_gpu(prob_each, len(test_loader) * world_size  * args.val_batch_size)
    pred_res = collect_results_gpu(pred_each, len(test_loader) * world_size  * args.val_batch_size)

    save_pickle(user_res, './user_res')
    save_pickle(item_res, './item_res')
    save_pickle(impress_res, './impress_res')
    save_pickle(truth_res, './truth_res')
    save_pickle(prob_res, './prob_res')
    save_pickle(pred_res, './pred_res')




        
def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        print('distributed')
        import datetime
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=6000))

    print(f'Building test loader at GPU {gpu}')

    test_task_list = {'sequential': ['1-1']}
    test_sample_numbers = {'sequential': 1}
    test_loader = get_loader(
        args,
        test_task_list,
        test_sample_numbers,
        split=args.test,
        mode='val',
        batch_size=args.val_batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )
    print(f'Building test loader at GPU {gpu}', len(test_loader))
    evaluate(test_loader)


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node



    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)  # only care about sequential loss

    LOSSES_NAME.append('pair_loss')
    LOSSES_NAME.append('total_loss')
    args.LOSSES_NAME = LOSSES_NAME




    comments = []
    dsets = []
    if 'MIND' in args.train:
        dsets.append('MIND')

    comments.append(''.join(dsets))

    args.backbone = 't5-small'
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))

    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M')

    #project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
