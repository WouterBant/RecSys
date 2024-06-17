from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from prompt_templates import create_prompt_titles, create_prompt_subtitles, create_prompt_qa_fast
import torch

class EkstraBladetDataset(Dataset):

    def __init__(self, args, create_prompt, split="train", debug=False):
        self.split = split
        
        # Download the dataset from huggingface
        if split == "hoe":
            split = "train"
        if split == "test":
            self.behaviors = load_dataset(f'Wouter01/testbehaviors', cache_dir=f"../../testbehaviors_data")["train"]
            self.articles = load_dataset(f'Wouter01/testarticles', cache_dir=f"../../testarticles_data")["train"].to_pandas()
            self.history = load_dataset(f'Wouter01/testhistory', cache_dir=f"../../testhistory_data")["train"].to_pandas()
        else:
            if args.evaltrain:
                self.behaviors = load_dataset(f'Wouter01/RecSys_{args.dataset}', 'behaviors', cache_dir=f"../../{args.dataset}_data")["train"]
                self.articles = load_dataset(f'Wouter01/RecSys_{args.dataset}', 'articles', cache_dir=f"../../{args.dataset}_data")["train"].to_pandas()
                self.history = load_dataset(f'Wouter01/RecSys_{args.dataset}', 'history', cache_dir=f"../../{args.dataset}_data")["train"].to_pandas()
            else:
                self.behaviors = load_dataset(f'Wouter01/RecSys_{args.dataset}', 'behaviors', cache_dir=f"../../{args.dataset}_data")[split]
                self.articles = load_dataset(f'Wouter01/RecSys_{args.dataset}', 'articles', cache_dir=f"../../{args.dataset}_data")["train"].to_pandas()
                self.history = load_dataset(f'Wouter01/RecSys_{args.dataset}', 'history', cache_dir=f"../../{args.dataset}_data")[split].to_pandas()

        # Set fast lookup for identifier keys
        self.history.set_index("user_id", inplace=True)
        self.articles.set_index("article_id", inplace=True)

        self.T = args.T  # Number of previous clicked articles to consider
        self.create_prompt = create_prompt  # Function to create a prompt from the data
        self.debug = debug
        self.datafraction = args.datafraction
        if args.model == "QA+" and split == "validation":
            self.datafraction = 0.0005
        self.args = args

    def __len__(self):
        return int(self.datafraction*len(self.behaviors))

    def __getitem__(self, idx):
        # Every item consits of a positive and negative sample
        behavior = self.behaviors[idx]

        # Get the impression id for gerenating the submission file
        impression_id = behavior["impression_id"]

        # Get the inview articles
        inview_articles = behavior["article_ids_inview"]

        # Get the history of the user
        user_id = behavior["user_id"]
        history = self.history.loc[user_id]

        # Get the T latest clicked articles by the user
        old_clicks = history["article_id_fixed"]
        old_clicks = old_clicks[-min(len(old_clicks), self.T):]

        if self.args.model == "QA+":
            titles_clicked = []
            for article_id in old_clicks.tolist():
                article = self.articles.loc[article_id]
                titles_clicked.append(article["title"])

            random.shuffle(inview_articles)
            titles_inview = []
            for article_id in inview_articles:
                article = self.articles.loc[article_id]
                titles_inview.append(article["title"])
            
            targets = [0]*len(inview_articles)
            target_idx = -1
            for idx, i in enumerate(inview_articles):
                if i in behavior["article_ids_clicked"]:
                    target_idx = idx
                    targets[idx] = 1
                    break
            assert target_idx != -1

            return {
                "prompt": self.create_prompt(titles_clicked),
                "decoder_input": " > @ ".join(titles_inview)[5:] + " > @ " ,
                "target": target_idx,
                "target_one_hot": targets
            }

        if self.split == "validation" or self.split ==  "hoe":
            # Get the past article information
            titles, subtitles, categories = [], [], []
            for article_id in old_clicks.tolist():
                article = self.articles.loc[article_id]
                titles.append(article["title"])
                subtitles.append(article["subtitle"])
                categories.append(article["category_str"])
            
            prompts = []
            for article_id in inview_articles:
                article = self.articles.loc[article_id]
                prompts.append(self.create_prompt(titles, subtitles, article["title"], article["subtitle"]))
                categories.append(article["category_str"])
            
            clicked_articles = behavior["article_ids_clicked"]
            targets = [1 if article in clicked_articles else 0 for article in inview_articles]
            return prompts, targets, categories

        if self.split == "test":
            # Get the past article information
            titles, subtitles = [], []
            for article_id in old_clicks.tolist():
                article = self.articles.loc[article_id]
                titles.append(article["title"])
                subtitles.append(article["subtitle"])
            
            prompts = []
            for article_id in inview_articles:
                article = self.articles.loc[article_id]
                prompts.append(self.create_prompt(titles, subtitles, article["title"], article["subtitle"]))
            return prompts, impression_id

        # Pick random positive and negative samples
        clicked_articles = behavior["article_ids_clicked"]
        unclicked_articles = [article for article in inview_articles if article not in clicked_articles]
        pos_sample = random.choice(clicked_articles)

        # If all documents clicked just treat one of them as a negative sample
        if len(unclicked_articles) == 0:
            unclicked_articles = clicked_articles
        neg_sample = random.choice(unclicked_articles)

        # Get the article information
        titles, subtitles = [], [] # last two are pos and neg samples
        for article_id in old_clicks.tolist() + [pos_sample, neg_sample]:
            article = self.articles.loc[article_id]
            titles.append(article["title"])
            subtitles.append(article["subtitle"])

        assert len(titles) == self.T + 2 and len(titles) == len(subtitles)
        
        title_pos, title_neg = titles[-2], titles[-1]
        subtitle_pos, subtitle_neg = subtitles[-2], subtitles[-1]
        titles, subtitles = titles[:-2], subtitles[:-2]

        # Create the prompts
        pos_prompt = self.create_prompt(titles, subtitles, title_pos, subtitle_pos)
        neg_prompt = self.create_prompt(titles, subtitles, title_neg, subtitle_neg)

        if self.debug:
            print("idx", idx)
            print("behavior", behavior)
            print("history", history)
            print("pos_prompt", pos_prompt)
            print("neg_prompt", neg_prompt)

        return pos_prompt, neg_prompt
    
class CollatorTrain:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pos_inputs = self.tokenizer([p for p, _ in batch], return_tensors='pt', padding=True, truncation=True)
        neg_inputs = self.tokenizer([n for _, n in batch], return_tensors='pt', padding=True, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            pos_targets = self.tokenizer(['ja' for _ in batch], return_tensors="pt", padding=True, truncation=True)
            neg_targets = self.tokenizer(['nej' for _ in batch], return_tensors="pt", padding=True, truncation=True)
            decoder_start = self.tokenizer(['ja / nej' for _ in batch], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "pos_input_ids": pos_inputs["input_ids"],
            "pos_attention_mask": pos_inputs["attention_mask"],
            "neg_input_ids": neg_inputs["input_ids"],
            "neg_attention_mask": neg_inputs["attention_mask"],
            "pos_labels": pos_targets["input_ids"],
            "neg_labels": neg_targets["input_ids"],
            "decoder_start": decoder_start["input_ids"],
        }
    
class CollatorValidation:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # TODO now relies on batch size 1
        prompts = self.tokenizer([p for prompts, _, _ in batch for p in prompts], return_tensors='pt', padding=True, truncation=True)
        targets = batch[0][1]
        categories = batch[0][2]
        
        with self.tokenizer.as_target_tokenizer():
            decoder_start = self.tokenizer(['ja / nej' for prompts, _, _ in batch for p in prompts], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "prompt_input_ids": prompts["input_ids"],
            "prompt_attention_mask": prompts["attention_mask"],
            "decoder_start": decoder_start["input_ids"],
            "targets": targets,
            "categories": categories  # can be used for diversity evaluation
        }

class CollatorTest:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompts = self.tokenizer([p for prompts, _ in batch for p in prompts], return_tensors='pt', padding=True, truncation=True)
        impression_ids = [i for prompts, i in batch for p in prompts]
        
        with self.tokenizer.as_target_tokenizer():
            decoder_start = self.tokenizer(['ja / nej' for prompts, _ in batch for p in prompts], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "prompt_input_ids": prompts["input_ids"],
            "prompt_attention_mask": prompts["attention_mask"],
            "decoder_start": decoder_start["input_ids"],
            "impression_ids": impression_ids
        }

class CollatorQAfast:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        inputs = self.tokenizer([b["prompt"] for b in batch], return_tensors='pt', padding=True, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            decoder_start = self.tokenizer([b["decoder_input"] for b in batch], return_tensors="pt", padding=True, truncation=True)

        target_idxs = torch.tensor([b["target"] for b in batch])

        # Create a tensor to hold the positions with the same shape as targets
        positions_tensor = torch.full(target_idxs.shape, -1, dtype=torch.long)

        # Iterate over each row to find positions of 1250
        for i in range(target_idxs.size(0)):
            row = decoder_start["input_ids"][i]
            target_row = target_idxs[i]
            
            for idx, j in enumerate(row):
                if j == 1250:
                    target_row -= 1
                    if target_row == -1:
                        positions_tensor[i] = idx
                        break
            
        return {
            "pos_input_ids": inputs["input_ids"],
            "pos_attention_mask": inputs["attention_mask"],
            "decoder_start": decoder_start["input_ids"],
            "targets_idxs": positions_tensor,
            "targets_one_hot": batch[0]["target_one_hot"],
        }

def get_loader(args, split, tokenizer, debug=False):
    assert split in ['train', 'validation', 'test'], 'dataset should be one of train, dev, test'

    if args.model == "QA+":
        create_prompt = create_prompt_qa_fast
        collator = CollatorQAfast(tokenizer)
        data = EkstraBladetDataset(args, create_prompt, split=split)  # todo change this to split
        shuffle = (split == "train")
        bs = args.batch_size if split == "train" else 1
        return DataLoader(data, batch_size=bs, collate_fn=collator, num_workers=args.num_workers, shuffle=shuffle)

    if args.titles:
        create_prompt = create_prompt_titles
    else:
        print("hi")
        create_prompt = create_prompt_subtitles

    if split == "train":
        print("hallo")
        collator = CollatorTrain(tokenizer)
        data = EkstraBladetDataset(args, create_prompt, split=split, debug=debug)
        return DataLoader(data, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers, shuffle=True)
    if split == "validation":
        collator = CollatorValidation(tokenizer)
        data = EkstraBladetDataset(args, create_prompt, split=split, debug=debug)  # TODO je kan deze naar hoe aanpassen om te testen op train set
        return DataLoader(data, batch_size=1, collate_fn=collator, num_workers=args.num_workers, shuffle=False)
    if split == "test":
        collator = CollatorTest(tokenizer)
        data = EkstraBladetDataset(args, create_prompt, split=split, debug=debug)
        return DataLoader(data, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers, shuffle=False)
    
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    collator = CollatorTrain(tokenizer)
    train_loader = get_loader('large', 'train', collator, T=5, debug=True)

    import time 
    num_samples = 1
    start_time = time.time()
    for batch in train_loader:
        # print(batch)
        num_samples -= 1
        if num_samples == 0:
            break
    end_time = time.time()
    print(f'Number of sampels/sec = {50*64/(end_time-start_time)}')