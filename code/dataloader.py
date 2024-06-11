from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from prompt_templates import create_prompt


class EkstraBladetDataset(Dataset):

    def __init__(self, args, create_prompt, split="train", debug=False):
        
        # Download the dataset from huggingface
        if split == "test":
            self.behaviors = load_dataset(f'Wouter01/testbehaviors', cache_dir=f"../../testbehaviors_data")["train"]
            self.articles = load_dataset(f'Wouter01/testarticles', cache_dir=f"../../testarticles_data")["train"].to_pandas()
            self.history = load_dataset(f'Wouter01/testhistory', cache_dir=f"../../testhistory_data")["train"].to_pandas()
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
        self.split = split

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
        titles, subtitles = [], []  # last two are pos and neg samples
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

def get_loader(args, split, tokenizer, debug=False):
    assert split in ['train', 'validation', 'test'], 'dataset should be one of train, dev, test'

    if split == "test":
        collator = CollatorTest(tokenizer)
        data = EkstraBladetDataset(args, create_prompt, split=split, debug=debug)
        return DataLoader(data, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers, shuffle=False)
    
    assert args.dataset in ['demo', 'large'], 'dataset should be one of demo, large'
    
    collator = CollatorTrain(tokenizer)
    data = EkstraBladetDataset(args, create_prompt, split=split, debug=debug)
    return DataLoader(data, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers, shuffle=True)


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