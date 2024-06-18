import torch
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer
import os
import copy
import wandb
from tqdm import tqdm

from evaluate import evaluate
from data.dataloader import get_loader
from utils.scheduler import CosineWarmupScheduler
from utils.argparser import argparser
from models.get_model import get_model


def train(args):
    model = get_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader_train = get_loader(args, 'train', tokenizer)
    data_loader_val = get_loader(args, 'validation', tokenizer)

    scheduler = CosineWarmupScheduler(optimizer, max_lr=args.lr, warmup_steps=args.warmup_steps, total_steps=len(data_loader_train) * args.n_epochs)
    scheduler.current_step = args.current_step

    best_metric, best_model = 0, None
    scaler = GradScaler()
    best_mrr = 0
    n_steps = 0
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        total_loss = 0

        for batch in tqdm(data_loader_train):
            
            # Checkpointing every 2000 steps
            if n_steps % 2000 == 0:
                torch.save(model.state_dict(), f"../checkpoints/model_lr_{args.lr}_lab_{args.labda}_model_{args.model}_prompt_{args.prompt}.pth")

                # Also save the best model on the validation set
                results = evaluate(args, model, data_loader_val)
                if results["mrr"] > best_mrr:
                    best_mrr = results["mrr"]
                    torch.save(model.state_dict(), f"../checkpoints/bestmodel_lr_{args.lr}_lab_{args.labda}_model_{args.model}_prompt_{args.prompt}.pth")

            n_steps += 1

            # Forward pass
            loss, pos_prob_yes, neg_prob_yes = model.train_step(batch)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Update parameters
            scaler.step(optimizer)
            scaler.update()

            # Update lr
            cur_lr = scheduler.step()
            total_loss += loss.item()

            if args.use_wandb:
                accuracy = (pos_prob_yes > neg_prob_yes).float().mean()  # Same two used for pair wise rank loss
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_accuracy': accuracy.item(),
                    'lr': cur_lr,
                    'avg_pos_prob_yes': pos_prob_yes.mean(),
                    'avg_neg_prob_yes': neg_prob_yes.mean(),
                })
    
    final_model = copy.deepcopy(model.state_dict())
    return results, final_model, best_model

if __name__ == '__main__':
    args = argparser()

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = '26de9d19e20ea7e7f7352e5b36f139df8d145bc8'  # TODO fill this in
        wandb.init(
            project=f"train_{args.backbone.split('/')[1]}_{args.dataset}_n_epochs_{args.n_epochs}_lr_{args.lr}_lab_{args.labda}__model_{args.model}_prompt_{args.prompt}",
            group=f"{args.backbone}",
            entity="RecSysPGNR",
        )

    results, final_model, best_model = train(args)