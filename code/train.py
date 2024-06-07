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

def compute_rank_loss(logits_pos, logits_neg):
    r_pos = torch.sigmoid(logits_pos)
    r_neg = torch.sigmoid(logits_neg)
    diff = torch.sigmoid(r_pos - r_neg)
    return torch.log(1e-8 + torch.exp(diff))



def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data_loader = get_loader('train')
    ce = CrossEntropyLoss()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=128)


    best_metric, best_model = 0, None

    for epoch in tqdm(range(args.n_epochs)):
        model.train()

        # training iteration
        total_loss = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            # Make sure everything is put to device

            input_pos, input_neg = batch
            input_pos = input_pos.to(device)
            input_neg = input_neg.to(device)
            decoder_input = tokenizer(["",""], return_tensors="pt")

            outputs_pos = model.base_model(input_ids=input_pos.input_ids, 
                                           attention_mask=input_pos.attention_mask,
                                           decoder_input_ids=decoder_input.input_ids,
                                           decoder_attention_mask=decoder_input.attention_mask)
            outputs_neg = model.base_model(input_ids=input_neg.input_ids, 
                                           attention_mask=input_neg.attention_mask,
                                           decoder_input_ids=decoder_input.input_ids,
                                           decoder_attention_mask=decoder_input.attention_mask)

            #logits = [yes,no] (magic numbers = token idxs for 'yes' and 'no')
            logits_pos = torch.stack((outputs_pos.logits[:,-1,36399], outputs_pos.logits[:,-1,375]), dim=1)
            logits_neg = torch.stack((outputs_neg.logits[:,-1,36399], outputs_neg.logits[:,-1,375]), dim=1)

            target_pos = torch.tensor([1,0],dtype=torch.float).unsqueeze(0).repeat(logits_pos.shape[0],1)
            target_neg = torch.tensor([0,1],dtype=torch.float).unsqueeze(0).repeat(logits_pos.shape[0],1)

            loss_nll = ce(logits_pos, target_pos) + ce(logits_neg, target_neg)
            loss_bpr = -compute_rank_loss(logits_pos[0], logits_neg[0]).mean(dim=0)

            lamb=0.5
            loss = (1-lamb)*loss_nll + lamb*loss_bpr

            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            if args.use_wandb and args.debug:
                wandb.log({'batch_loss': loss.item()})

        if args.use_wandb:
            wandb.log({'epoch_loss': total_loss / len(data_loader)})
        
        # validation
        if (epoch + 1) % args.eval_interval == 0:
            results = evaluate(model, 'dev')

            if args.use_wandb:
                wandb.log(results)  # this will not work but do this for all metrics
            
            # TODO fix this, just ndcg or someting
            if results['metric'] > best_metric:
                best_metric = results['metric']
                best_model = copy.deepcopy(model.state_dict())
    
    # test
    model = model.load_state_dict(best_model)
    model.eval()
    results = evaluate(model, 'test')
    if args.use_wandb:
        wandb.log(results)  # TODO fix this + say it is for test set

    final_model = copy.deepcopy(model.state_dict())
    return results, final_model, best_model


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='t5-small', help='backbone model')
    parser.add_argument('--tokenizer', type=str, default='t5-small', help='tokenizer model')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint to pretrained model')
    parser.add_argument('--labda', type=float, default=0.5, help='lambda for pairwise ranking loss')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('use_wandb', type=bool, default=False, help='use wandb for logging') 
    parser.add_argument('debug', type=bool, default=False, help='debug mode')
    parser.add_argument('dataset', type=str, default='train', help='dataset to train on')
    parser.add_argument('eval_interval', type=int, default=1, help='evaluate model every n epochs')
    parser.add_argument('from_checkpoint', type=str, default='', help='load model from checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()
    print(args)

    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = '26de9d19e20ea7e7f7352e5b36f139df8d145bc8'  # TODO fill this in
        wandb.init(
            project=f"{args.backbone}_{args.dataset}_{args.lr}_{args.n_epochs}",
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