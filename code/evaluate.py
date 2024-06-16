import argparse
from models.get_model import get_model
import torch
from collections import defaultdict
import os
from tqdm import tqdm
from dataloader import get_loader
import wandb
from metrics import MetricsEvaluator
from transformers import AutoTokenizer
import numpy as np


def evaluate(args, model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(int)

    metrics_evaluator = MetricsEvaluator(k=5)

    model.to(device)
    model.eval()
    total = 0

    for batch in tqdm(data_loader):
        probs = model.validation_step(batch)

        # Collect data for metric evaluation
        scores = probs.squeeze().cpu().numpy()
        if args.model == "QA+":
            labels = np.array(batch["targets_one_hot"])
        else:
            labels = np.array(batch["targets"])
        # recommendations = batch["recommendations"].cpu().numpy().tolist()
        # candidate_items = batch.get("candidate_items", []).cpu().numpy().tolist()  # Optional
        # click_histories = batch.get("click_histories", []).cpu().numpy().tolist()  # Optional

        metrics = metrics_evaluator.compute_metrics({
            'scores': scores,
            'labels': labels,
            # 'recommendations': recommendations,
            # 'candidate_items': candidate_items,
            # 'click_histories': click_histories
        })

        for key in metrics:
            results[key] += metrics[key]*len(scores)
        total += len(scores)
    
    for key in results:
        results[key] /= total
    
    # If using wandb, log the metrics
    if args.use_wandb:
        wandb.log(results)
  
    return results


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='google/mt5-small', help='backbone model')
    parser.add_argument('--tokenizer', type=str, default='google/mt5-small', help='tokenizer model')
    parser.add_argument('--labda', type=float, default=0.5, help='lambda for pairwise ranking loss')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--current_step', type=int, default=0, help='starting step for cosine learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--titles', action='store_true', help='use titles instead of subtitles in prompt')
    parser.add_argument('--use_classifier', action='store_true', help='use classifier on top of positive logits')
    parser.add_argument('--use_QA_model', action='store_true', help='use QA model instead of generative model')
    parser.add_argument('--T', type=int, default=4, help='number of previous clicked articles to include in the prompt')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset to train on')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluate model every n epochs')
    parser.add_argument('--from_checkpoint', type=str, default='', help='load model from checkpoint')
    parser.add_argument('--datafraction', type=float, default=1.0, help='fraction of data to use')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = '26de9d19e20ea7e7f7352e5b36f139df8d145bc8'  # TODO fill this in
        wandb.init(
            project=f"eval_{args.backbone.split('/')[1]}_{args.dataset}_{args.lr}_{args.n_epochs}",
            group=f"{args.backbone}",
            entity="RecSysPGNR",
        )

    model = get_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader_val = get_loader(args, 'validation', tokenizer)
    results = evaluate(args, model, data_loader_val)
