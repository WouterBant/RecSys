import argparse
from models import get_model
from dataloader import get_loader
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer


def generate_and_write_predictions(args, output_filename="predictions.txt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader = get_loader(args, 'test', tokenizer)
    model.eval()

    previous_impression_id = None
    impression_probs = []
    
    with open(output_filename, 'w') as f, torch.no_grad():

        for batch in tqdm(data_loader):
            impression_id = batch["impression_ids"]
            input_ids = batch["prompt_input_ids"].to(device)
            attention_mask = batch["prompt_attention_mask"].to(device)
            decoder_input_ids = batch["decoder_start"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
            logits = outputs.logits[:,0,:]
            prob_yes = torch.softmax(logits, dim=-1)[:, 432]
            prob_yes = prob_yes.tolist()

            assert len(logits) == len(impression_id), print(len(logits), len(impression_id))
            
            for p, i in zip(prob_yes, impression_id):
                if previous_impression_id != i:
                    if previous_impression_id is not None:
                        # Write previous impression's predictions
                        sorted_idxs = np.argsort(np.array(impression_probs))+1
                        f.write(f"{previous_impression_id} {sorted_idxs}\n")

                    # Reset impression_logits and previous_impression_id
                    impression_probs = []
                    previous_impression_id = impression_id

                impression_probs.append(p)
        
        # Don't forget to write the last impression's predictions
        sorted_idxs = np.argsort(np.array(impression_probs))
        f.write(f"{previous_impression_id}:\t{sorted_idxs}\n")


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='google/mt5-small', help='backbone model')
    parser.add_argument('--tokenizer', type=str, default='google/mt5-small', help='tokenizer model')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint to pretrained model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--use_QA_model', action='store_true', help='use QA model instead of generative model')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--T', type=int, default=4, help='number of previous clicked articles to include in the prompt')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset to train on')
    parser.add_argument('--datafraction', type=float, default=1.0, help='fraction of data to use')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argparser()
    print(args)
    generate_and_write_predictions(args, output_filename="predictions.txt")