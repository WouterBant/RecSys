import torch
from transformers import AutoTokenizer
import os
import json
import wandb
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from data.dataloader import get_loader
from models.get_model import get_model
from utils.argparser import argparser
from utils.metrics import MetricsEvaluator


def evaluate(args, model, data_loader):  # TODO only capable of batch size 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(int)

    metrics_evaluator = MetricsEvaluator(k=5, T=args.T)

    model.to(device)
    model.eval()

    # Count number of examples in dataset
    total = 0

    for batch in tqdm(data_loader):

        # Forward pass for evaluation
        probs = model.validation_step(batch)

        # Collect data for metric evaluation
        scores = probs.squeeze().cpu().numpy()
        labels = np.array(batch["targets"])
        categories = np.array(batch["categories"])

        metrics = metrics_evaluator.compute_metrics({
            'scores': scores,
            'labels': labels,
            'categories': categories,
        })

        # Sum the metrics
        for key in metrics:
            results[key] += metrics[key]*len(scores)
        total += len(scores)
    
    # Take the average of the metrics over the dataset
    for key in results:
        results[key] /= total
    
    # If using wandb, log the metrics (bit useless, in a second will be written to a file anyway)
    if args.use_wandb:
        wandb.log(results)
  
    return results


if __name__ == '__main__':
    args = argparser()

    name = "train" if args.evaltrain else "validation"

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = '26de9d19e20ea7e7f7352e5b36f139df8d145bc8'  # TODO fill this in
        wandb.init(
            project=f"eval_{name}_{args.from_checkpoint.split('/')[-1]}",
            group=f"{args.backbone}",
            entity="RecSysPGNR",
        )

    model = get_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader_val = get_loader(args, 'validation', tokenizer)
    results = evaluate(args, model, data_loader_val)

    with open(f"../results/{name}/{args.from_checkpoint.split('/')[-1][:-4]}.json", 'w') as file:
        json.dump(results, file)