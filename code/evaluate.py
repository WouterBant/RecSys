import argparse
import json
from models import get_model
from metrics import compute_metrics
from dataloader import get_loader
import torch
from collections import defaultdict
from datetime import datetime
import os
from utils import compute_rank_loss
from torch.nn import CrossEntropyLoss


def evaluate(args, model, tokenizer, T, split='validation'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(list)

    data_loader = get_loader(args, split, tokenizer, T, debug=False)
    ce = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.to(device)
    model.eval()

    for batch in data_loader:
        # Forward pass for the positive and negative examples
        pos_outputs = model(
            input_ids=batch["pos_input_ids"].to(device), 
            attention_mask=batch["pos_attention_mask"].to(device),
            labels=batch["pos_labels"].to(device),
        )
        neg_outputs = model(
            input_ids=batch["neg_input_ids"].to(device),
            attention_mask=batch["neg_attention_mask"].to(device),
            labels=batch["neg_labels"].to(device),
        )

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
        accuracy = (pos_prob_yes > neg_prob_yes).float().mean()
        results["loss_nll"].append(loss_nll.item())
        results["loss_bpr"].append(loss_bpr.item())
        results["loss"].append(loss.item())
        results["accuracy"].append(accuracy.item())

    # Convert lists in results to tensors
    for key in results:
        results[key] = torch.Tensor(results[key])

    # Compute mean and standard deviation
    mean_results = {}
    std_results = {}
    for key, value in results.items():
        mean_results[key] = torch.mean(value).item()
        std_results['std_'+key] = torch.std(value).item()
    
    return mean_results, std_results


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