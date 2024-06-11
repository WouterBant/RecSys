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
from tqdm import tqdm


def evaluate(args, model, tokenizer, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = defaultdict(int)

    ce = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.to(device)
    model.eval()
    total = 0

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
    
    for key in results:
        results[key] /= total
    
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