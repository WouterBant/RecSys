import torch
from transformers import AutoTokenizer
import random
import numpy as np
from tqdm import tqdm

from data.dataloader import get_loader
from models.get_model import get_model
from utils.argparser import argparser


def generate_and_write_predictions(args, output_filename="predictions.txt"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("getting_model")
    model = get_model(args).to(device)
    print("hi")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_loader = get_loader(args, 'test', tokenizer)
    model.eval()

    previous_impression_id = None
    impression_probs = []
    
    with open(output_filename, 'w') as f:

        for batch in tqdm(data_loader):
            impression_id = batch["impression_ids"]
            input_ids = batch["prompt_input_ids"].to(device)
            attention_mask = batch["prompt_attention_mask"].to(device)
            decoder_input_ids = batch["decoder_start"].to(device)

            # Process the batch in chunks of at most 16
            for start_idx in range(0, len(impression_id), 16):
                end_idx = min(start_idx + 16, len(impression_id))
                chunk_impression_id = impression_id[start_idx:end_idx]

                chunk = {
                    "prompt_input_ids": input_ids[start_idx:end_idx],
                    "prompt_attention_mask": attention_mask[start_idx:end_idx],
                    "decoder_start": decoder_input_ids[start_idx:end_idx]
                }
                
                with torch.no_grad():
                    outputs = model.validation_step(chunk)

                prob_yes = outputs.tolist()
                for p, i in zip(prob_yes, chunk_impression_id):
                    if previous_impression_id != i:
                        if previous_impression_id is not None:
                            # Write previous impression's predictions
                            sorted_idxs = np.argsort(np.array(impression_probs)) + 1
                            print(sorted_idxs, )
                            f.write(f"{previous_impression_id} [{','.join(map(str, sorted_idxs.tolist()))}]\n")
                            f.flush()

                        # Reset impression_logits and previous_impression_id
                        impression_probs = []
                        previous_impression_id = i

                    impression_probs.append(p)

        # Don't forget to write the last impression's predictions
        sorted_idxs = np.argsort(np.array(impression_probs))
        f.write(f"{previous_impression_id} [{','.join(map(str, sorted_idxs.tolist()))}]\n")


if __name__ == '__main__':
    args = argparser()
    generate_and_write_predictions(args, output_filename="predictions.txt")