import argparse
from datetime import datetime
from models.get_model import get_model
from dataloader import get_loader
import torch
from evaluate import evaluate
import copy
import wandb
from tqdm import tqdm
import os
import json
from scheduler import CosineWarmupScheduler
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer


def train(args):
    model = get_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader_train = get_loader(args, 'train', tokenizer)
    data_loader_val = get_loader(args, 'validation', tokenizer)

    if args.use_wandb:
        wandb.log({
            "T": args.T,
            "lambda": args.labda,
            "model": args.model,
            "titles": float(int(args.titles==True)),
        })

    scheduler = CosineWarmupScheduler(optimizer, max_lr=args.lr, warmup_steps=args.warmup_steps, total_steps=len(data_loader_train) * args.n_epochs)
    scheduler.current_step = args.current_step

    best_metric, best_model = 0, None
    scaler = GradScaler()

    n_steps = 0
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        total_loss = 0

        for batch in tqdm(data_loader_train):
            
            # Checkpointing every 1000 steps
            if n_steps % 1000 == 0:
                torch.save(model.state_dict(), f"checkpoints/model_lr_{args.lr}_lab_{args.labda}_model_{args.model}_tit_{args.titles}.pth")
            n_steps += 1

            # Forward pass
            loss, pos_prob_yes, neg_prob_yes = model.train_step(batch)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_lr = scheduler.step()
            total_loss += loss.item()

            if args.use_wandb and args.debug:
                accuracy = (pos_prob_yes > neg_prob_yes).float().mean()
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_accuracy': accuracy.item(),
                    'lr': cur_lr,
                    'avg_pos_prob_yes': pos_prob_yes.mean(),
                    'avg_neg_prob_yes': neg_prob_yes.mean(),
                })

        if args.use_wandb:
            wandb.log({'epoch_loss': total_loss / len(data_loader_train)})
        
        # validation
        if (epoch + 1) % args.eval_interval == 0:
            results = evaluate(args, model, data_loader_val)

            if args.use_wandb:
                wandb.log(results)

            # TODO fix this, just ndcg or someting
            if results['accuracy'] > best_metric:
                best_metric = results['accuracy']
                best_model = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
    
    final_model = copy.deepcopy(model.state_dict())
    return results, final_model, best_model


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
    parser.add_argument('--titles', action='store_true', help='use titles instead of subtitles in prompt')
    parser.add_argument('--T', type=int, default=4, help='number of previous clicked articles to include in the prompt')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset to train on')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluate model every n epochs')
    parser.add_argument('--from_checkpoint', type=str, default='', help='load model from checkpoint')
    parser.add_argument('--datafraction', type=float, default=1.0, help='fraction of data to use')
    parser.add_argument('--warmup_steps', type=int, default=15000, help='number of warmup steps')
    parser.add_argument('--model', type=str, choices=["QA", "QA+", "CG"], help='model to train')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()
    print(args)

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = '26de9d19e20ea7e7f7352e5b36f139df8d145bc8'  # TODO fill this in
        wandb.init(
            project=f"{args.backbone.split('/')[1]}_{args.dataset}_n_epochs_{args.n_epochs}_lr_{args.lr}_lab_{args.labda}__model_{args.model}_tit_{args.titles}",
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