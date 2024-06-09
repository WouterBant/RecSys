import argparse
from datetime import datetime
from models import get_model
from dataloader import get_loader
import torch
from evaluate import evaluate
import copy
import wandb
from tqdm import tqdm
import os
import json
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from scheduler import CosineWarmupScheduler
from utils import compute_rank_loss


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader = get_loader(args, 'train', tokenizer, T=args.T, debug=False)

    if args.use_wandb:
        # TODO add more
        wandb.log({
            "T": args.T,
            "lambda": args.labda,
        })

    # TODO fix the hardcoding here
    scheduler = CosineWarmupScheduler(optimizer, max_lr=args.lr, warmup_steps=500, total_steps=len(data_loader) * args.n_epochs)
    ce = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_metric, best_model = 0, None

    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        total_loss = 0

        for batch in tqdm(data_loader):

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

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()
            cur_lr = scheduler.step()
            total_loss += loss.item()

            if args.use_wandb and args.debug:
                accuracy = (pos_prob_yes > neg_prob_yes).float().mean()
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_accuracy': accuracy,
                    'lr': cur_lr
                })

        if args.use_wandb:
            wandb.log({'epoch_loss': total_loss / len(data_loader)})
        
        # validation
        if (epoch + 1) % args.eval_interval == 0:
            mean_results, std_results = evaluate(args, model, tokenizer, args.T, 'validation')

            if args.use_wandb:
                wandb.log(mean_results)
                wandb.log(std_results)
            
            # TODO fix this, just ndcg or someting
            if results['metric'] > best_metric:
                best_metric = mean_results['accuracy']
                best_model = copy.deepcopy(model.state_dict())
    
    # test
    model.load_state_dict(best_model)
    model.eval()
    results = evaluate(model, 'test')
    if args.use_wandb:
        wandb.log(results)  # TODO fix this + say it is for test set

    final_model = copy.deepcopy(model.state_dict())
    return results, final_model, best_model


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='google/mt5-small', help='backbone model')
    parser.add_argument('--tokenizer', type=str, default='google/mt5-small', help='tokenizer model')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint to pretrained model')
    parser.add_argument('--labda', type=float, default=0.5, help='lambda for pairwise ranking loss')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--T', type=int, default=4, help='number of previous clicked articles to include in the prompt')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset to train on')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluate model every n epochs')
    parser.add_argument('--from_checkpoint', type=str, default='', help='load model from checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()
    print(args)

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = '26de9d19e20ea7e7f7352e5b36f139df8d145bc8'  # TODO fill this in
        wandb.init(
            project=f"{args.backbone.split('/')[1]}_{args.dataset}_{args.lr}_{args.n_epochs}",
            group=f"{args.backbone}",
            entity="RecSysPGNR",
        )

    results, final_model, best_model = train(args)

    # save the final and best model + results on the test set
    time = datetime.now().strftime('%b%d_%H-%M')
    os.makedirs(f'/checkpoints/{time}', exist_ok=True)
    torch.save(final_model, f'checkpoints/{time}/final_model_lr{args.lr}_model{args.backbone}_epochs{args.n_epochs}.pth')
    torch.save(best_model, f'checkpoints/{time}/best_model_lr{args.lr}_model{args.backbone}_epochs{args.n_epochs}.pth')
    os.makedirs(f'/results/{time}', exist_ok=True)
    with open(f'/results/{time}/{args.model}_{args.labda}.json', 'w') as f:
        json.dump(results, f)