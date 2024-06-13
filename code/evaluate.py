import argparse
import json
from models import get_model
import torch
from collections import defaultdict
from datetime import datetime
import os
from utils import compute_rank_loss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer
from dataloader import get_loader
import wandb
from metrics import MetricsEvaluator


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(int)
    model = get_model(args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader = get_loader(args, 'test', tokenizer)

    ce = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    metrics_evaluator = MetricsEvaluator().compute_metrics

    model.to(device)
    model.eval()
    total = 0

    all_scores = []
    all_labels = []
    all_recommendations = []
    all_candidate_items = []  
    all_click_histories = []

    for batch in tqdm(data_loader):
        # Forward pass for the positive and negative examples
        with torch.no_grad():
            pos_outputs = model(
                input_ids=batch["pos_input_ids"].to(device), 
                attention_mask=batch["pos_attention_mask"].to(device),
                decoder_input_ids=batch["decoder_start"].to(device)
            )
            neg_outputs = model(
                input_ids=batch["neg_input_ids"].to(device),
                attention_mask=batch["neg_attention_mask"].to(device),
                decoder_input_ids=batch["decoder_start"].to(device)
            )

        if args.use_QA_model:
            pos_logits = pos_outputs.start_logits
            neg_logits = neg_outputs.start_logits
            pos_probs = torch.softmax(pos_logits, dim=-1)  # B, V -> B,V
            neg_probs = torch.softmax(neg_logits, dim=-1)
            batch_size = pos_probs.shape[0]
            pos_target = torch.tensor(batch_size * [0]).to(device) # 0 = idx of 'ja' token in decoder input sequence 
            neg_target = torch.tensor(batch_size * [3]).to(device) # 3 = idx of 'nej' token in decoder input sequence
            loss_nll = ce(pos_logits, pos_target) + ce(neg_logits, neg_target)
            pos_prob_yes = pos_probs[0]
            neg_prob_yes = neg_probs[0]
            loss_bpr = compute_rank_loss(pos_prob_yes, neg_prob_yes).mean(dim=0)
            loss = (1-args.labda)*loss_nll + args.labda*loss_bpr
        else:
            # Only take the first token (should be 'ja' or 'nej')
            pos_logits = pos_outputs.logits[:,0,:]  # B, T, V -> B, V
            neg_logits = neg_outputs.logits[:,0,:]

            # 36339 is the token id for 'ja'
            pos_prob_yes = torch.softmax(pos_logits, dim=-1)[:, 432]  # B, V -> B
            neg_prob_yes = torch.softmax(neg_logits, dim=-1)[:, 432]

            # Same for the targets that store one of the V labels
            pos_target = batch["pos_labels"][:,0].to(device)  # B, T -> B
            neg_target = batch["neg_labels"][:,0].to(device)

            # Compute loss
            loss_nll = ce(pos_logits, pos_target) + ce(neg_logits, neg_target)
            loss_bpr = compute_rank_loss(pos_prob_yes, neg_prob_yes).mean(dim=0)
            loss = (1-args.labda)*loss_nll + args.labda*loss_bpr

        accuracy = (pos_prob_yes > neg_prob_yes).float().sum()
        results["loss_nll"] += loss_nll.item()
        results["loss_bpr"] += loss_bpr.item()
        results["loss"] += loss.item()
        results["accuracy"] += accuracy.item() 
        total += batch["pos_input_ids"].size(0)

        # Collect data for metric evaluation
        scores = pos_prob_yes.cpu().numpy().tolist()
        labels = pos_target.cpu().numpy().tolist()
        recommendations = batch["recommendations"].cpu().numpy().tolist()
        candidate_items = batch.get("candidate_items", []).cpu().numpy().tolist()  # Optional
        click_histories = batch.get("click_histories", []).cpu().numpy().tolist()  # Optional

        all_scores.extend(scores)
        all_labels.extend(labels)
        all_recommendations.extend(recommendations)
        all_candidate_items.extend(candidate_items)  
        all_click_histories.extend(click_histories)  

    # Compute metrics
    output = {
        'scores': all_scores,
        'labels': all_labels,
        'recommendations': all_recommendations,
        'candidate_items': all_candidate_items,  # Optional
        'click_histories': all_click_histories  # Optional
    }

    metrics = metrics_evaluator.compute_metrics(output, lookup_dict=None, lookup_key=None)

    for key in results:
        results[key] /= total

    # If using wandb, log the metrics
    if args.use_wandb:
        wandb.log(metrics)
    
    return results


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='google/mt5-small', help='backbone model')
    parser.add_argument('--tokenizer', type=str, default='google/mt5-small', help='tokenizer model')
    parser.add_argument('--labda', type=float, default=0.5, help='lambda for pairwise ranking loss')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--current_step', type=int, default=0, help='starting step for cosine learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--debug', action='store_true', help='debug mode')
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
            project=f"{args.backbone.split('/')[1]}_{args.dataset}_{args.lr}_{args.n_epochs}",
            group=f"{args.backbone}",
            entity="RecSysPGNR",
        )

    results = evaluate(args)
