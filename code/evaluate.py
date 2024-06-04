# Implement the evaluation of the model
# This includes metrics, no abstraction needed, the metrics are all simple

import argparse
import json
from models import get_model
from metrics import compute_metrics
from dataloader import get_loader
import torch
from collections import defaultdict
from datetime import datetime
import os


def evaluate(model, dataset='dev'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(list)

    data_loader = get_loader(dataset)

    model.to(device)
    model.eval()
    for i, batch in enumerate(data_loader):
        # put everything to device
        output = model(batch)
        # compute metrics
        # results['metric'].append(compute_metric(output))  # finally len(results['metric']) == len(data_loader) so we can investigate variance
    return results

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='t5-small', help='backbone model')
    parser.add_argument('--tokenizer', type=str, default='t5-small', help='tokenizer model')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint to pretrained model')
    parser.add_argument('--labda', type=float, default=0.5, help='lambda for pairwise ranking loss')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()
    print(args)

    model = get_model(args)
    results = evaluate(model, 'test')

    time = datetime.now().strftime('%b%d_%H-%M')
    os.makedirs(f'/results/{time}', exist_ok=True)
    with open(f'/results/{time}/{args.model}_{args.labda}.json', 'w') as f:
        json.dump(results, f)