import argparse
from models import get_model
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
        input_ids = batch["prompt_input_ids"].to(device)
        attention_mask = batch["prompt_attention_mask"].to(device)
        decoder_input_ids = batch["decoder_start"].to(device)
        target = batch["targets"]

        # Forward pass for the positive and negative examples
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )

        if args.use_QA_model:
            # Only consider the start logits
            logits = outputs.start_logits

            # Only consider the probability for 'ja'
            probs = torch.softmax(logits, dim=-1)[:0]
        else:
            # Only take the first token (should be 'ja' or 'nej')
            logits = outputs.logits[:,0,:]  # B, T, V -> B, V

            # 36339 is the token id for 'ja'
            probs = torch.softmax(logits, dim=-1)[:, 432]  # B, V -> B
        
        # Collect data for metric evaluation
        scores = probs.cpu().numpy()
        labels = np.array(target)
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

        # If using wandb, log the metrics
        if args.use_wandb:
            wandb.log(metrics)

        for key in metrics:
            results[key] += metrics[key]*len(scores)
        total += len(scores)
    
    for key in results:
        results[key] /= total
  
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
