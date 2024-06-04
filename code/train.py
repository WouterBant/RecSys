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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    data_loader = get_loader('train')

    best_metric, best_model = 0, None

    for epoch in tqdm(range(args.n_epochs)):
        model.train()

        # training iteration
        total_loss = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            # Make sure everything is put to device
            output = model(batch)
            loss = output['loss']
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