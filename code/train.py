import argparse
from datetime import datetime
from models import get_model
from dataloader import get_loader
import torch
from torch import nn
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
from torch.cuda.amp import GradScaler


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader_train = get_loader(args, 'train', tokenizer)
    data_loader_val = get_loader(args, 'validation', tokenizer)

    if args.use_classifier:
        classifier = nn.Sequential(
            nn.Linear(250112, 512),
            nn.SiLU(),
            nn.Linear(512, 2)
        ).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr)
        ce_classifier = CrossEntropyLoss()

    if args.use_wandb:
        # TODO add more
        wandb.log({
            "T": args.T,
            "lambda": args.labda,
            "classifier": float(int(args.use_classifier==True)),
            "QA": float(int(args.use_QA_model==True)),
            "titles": float(int(args.titles==True)),
        })

    # TODO fix the hardcoding here
    scheduler = CosineWarmupScheduler(optimizer, max_lr=args.lr, warmup_steps=15000, total_steps=len(data_loader_train) * args.n_epochs)
    if args.use_QA_model:
        ce = CrossEntropyLoss()
    else:
        ce = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scheduler.current_step = args.current_step

    best_metric, best_model = 0, None
    scaler = GradScaler()

    n_steps = 0
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        total_loss = 0

        for batch in tqdm(data_loader_train):
            
            if n_steps % 1 == 0:
                torch.save(model.state_dict(), f"model_lr_{args.lr}_class_{args.use_classifier}_lab_{args.labda}__qa_{args.use_QA_model}_tit_{args.titles}.pth")
            n_steps += 1

            # Forward pass for the positive and negative examples
            pos_outputs = model(
                input_ids=batch["pos_input_ids"].to(device), 
                attention_mask=batch["pos_attention_mask"].to(device),
                decoder_input_ids=batch["decoder_start"].to(device),
            )
            neg_outputs = model(
                input_ids=batch["neg_input_ids"].to(device),
                attention_mask=batch["neg_attention_mask"].to(device),
                decoder_input_ids=batch["decoder_start"].to(device),
            )

            if args.use_QA_model:
                pos_logits = pos_outputs.start_logits
                neg_logits = neg_outputs.start_logits
                pos_probs = torch.softmax(pos_logits, dim=-1)  # B, V -> B,V
                neg_probs = torch.softmax(neg_logits, dim=-1)
                batch_size = pos_probs.shape[0]
                pos_target = torch.tensor(batch_size * [0]).to(device) # 0 = idx of 'ja' token in decoder input sequence 
                neg_target = torch.tensor(batch_size * [3]).to(device) # 3 = idx of 'nej' token in decoder input sequence
                # print(pos_logits, pos_target)
                loss_nll = ce(pos_logits, pos_target) + ce(neg_logits, neg_target)
                pos_prob_yes = pos_probs[0]
                neg_prob_yes = neg_probs[0]
                loss_bpr = compute_rank_loss(pos_prob_yes, neg_prob_yes).mean(dim=0)
                loss = (1-args.labda)*loss_nll + args.labda*loss_bpr
                # print(loss_nll, loss_bpr)
            else:
                # Only take the first token (should be 'ja' or 'nej')
                pos_logits = pos_outputs.logits[:,0,:]  # B, T, V -> B, V
                neg_logits = neg_outputs.logits[:,0,:]

                if args.use_classifier:
                    pos_classifier_logits = classifier(pos_logits)
                    neg_classifier_logits = classifier(neg_logits)

                    if args.use_wandb and args.debug:
                        wandb.log({
                            "pos_classifier_logits": pos_classifier_logits.mean(0)[0],
                            "neg_classifier_logits": neg_classifier_logits.mean(0)[0]
                        })

                    pos_labels = torch.ones(pos_classifier_logits.size(0), dtype=torch.long, device=device)  # All positive samples are labeled 1
                    neg_labels = torch.zeros(neg_classifier_logits.size(0), dtype=torch.long, device=device)  # All negative samples are labeled 0

                    loss_pos_classifier = ce_classifier(pos_classifier_logits, pos_labels)
                    loss_neg_classifier = ce_classifier(neg_classifier_logits, neg_labels)

                    pos_prob_yes = torch.softmax(pos_classifier_logits, dim=-1)[1]
                    neg_prob_yes = torch.softmax(neg_classifier_logits, dim=-1)[1]

                    loss_nll = loss_pos_classifier + loss_neg_classifier
                    loss_bpr = compute_rank_loss(pos_prob_yes, neg_prob_yes).mean(dim=0)
                    loss = (1-args.labda)*loss_nll + args.labda*loss_bpr

                else:
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
            results = evaluate(args, model, tokenizer, data_loader_val)

            if args.use_wandb:
                wandb.log(results)

            # TODO fix this, just ndcg or someting
            if results['accuracy'] > best_metric:
                best_metric = results['accuracy']
                best_model = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')
    
    # test
    model.load_state_dict(best_model)
    model.eval()
    # results = evaluate(model, 'test')
    # if args.use_wandb:
    #     wandb.log(results)  # TODO fix this + say it is for test set

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
    print(args)

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = '26de9d19e20ea7e7f7352e5b36f139df8d145bc8'  # TODO fill this in
        wandb.init(
            project=f"{args.backbone.split('/')[1]}_{args.dataset}_n_epochs_{args.n_epochs}_lr_{args.lr}_class_{args.use_classifier}_lab_{args.labda}__qa_{args.use_QA_model}_tit_{args.titles}",
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