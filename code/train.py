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

    scheduler = CosineWarmupScheduler(
        optimizer,
        max_lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=len(data_loader_train) * args.n_epochs
    )
    scheduler.current_step = args.current_step

    best_model = None
    scaler = GradScaler()
    best_mrr = n_steps = 0

    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        total_loss = 0

        for batch in tqdm(data_loader_train):

            # Checkpointing every 2000 steps
            if n_steps % 2000 == 0:
                file_path = (
                    f"../checkpoints/model_lr_{args.lr}_"
                    f"lab_{args.labda}_model_{args.model}_"
                    f"prompt_{args.prompt}.pth"
                )
                torch.save(model.state_dict(), file_path)

                # Also save the best model on the validation set
                results = evaluate(args, model, data_loader_val)
                if results["mrr"] > best_mrr:
                    best_mrr = results["mrr"]
                    best_model_path = (
                        f"../checkpoints/bestmodel_lr_{args.lr}_"
                        f"lab_{args.labda}_model_{args.model}_"
                        f"prompt_{args.prompt}.pth"
                    )
                    torch.save(model.state_dict(), best_model_path)

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
                # From pairwise rank loss (if available else 1)
                accuracy = (pos_prob_yes > neg_prob_yes).float().mean()

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
        try:
            os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
        except KeyError:
            print("Please set your WANBD_API_KEY as an environment variable")
            print("You can find your API key at: https://wandb.ai/authorize")
            print("You can set it as an environment variable using:")
            print("export WANDB_API_KEY='your_api_key'")
            exit(1)

        project = (
            f"train_{args.backbone.split('/')[1]}_"
            f"{args.dataset}_"
            f"n_epochs_{args.n_epochs}_"
            f"lr_{args.lr}_"
            f"lab_{args.labda}_"
            f"model_{args.model}_"
            f"prompt_{args.prompt}"
        )
        wandb.init(
            project=project,
            group=f"{args.backbone}",
            entity="RecSysPGNR",
        )

    results, final_model, best_model = train(args)
