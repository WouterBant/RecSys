from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from prompt_templates import create_prompt


class EkstraBladetDataset(Dataset):

    def __init__(self, create_prompt, dataset="demo", split="train", T=5, debug=False):
        
        # Download the dataset from huggingface
        self.behaviors = load_dataset(f'Wouter01/RecSys_{dataset}', 'behaviors', cache_dir=f"{dataset}_data")[split]
        self.articles = load_dataset(f'Wouter01/RecSys_{dataset}', 'articles', cache_dir=f"{dataset}_data")["train"].to_pandas()
        self.history = load_dataset(f'Wouter01/RecSys_{dataset}', 'history', cache_dir=f"{dataset}_data")[split].to_pandas()

        # Set fast lookup for identifier keys
        self.history.set_index("user_id", inplace=True)
        self.articles.set_index("article_id", inplace=True)

        self.T = T  # Number of previous clicked articles to consider
        self.create_prompt = create_prompt  # Function to create a prompt from the data
        self.debug = debug

    def __len__(self):
        return int(0.01*len(self.behaviors))

    def __getitem__(self, idx):
        # Every item consits of a positive and negative sample
        behavior = self.behaviors[idx]

        # Pick random positive and negative samples
        clicked_articles = behavior["article_ids_clicked"]
        unclicked_articles = [article for article in behavior["article_ids_inview"] if article not in clicked_articles]
        pos_sample = random.choice(clicked_articles)

        # If all documents clicked just treat one of them as a negative sample
        if len(unclicked_articles) == 0:
            unclicked_articles = clicked_articles
        neg_sample = random.choice(unclicked_articles)

        # Get the history of the user
        user_id = behavior["user_id"]
        history = self.history.loc[user_id]

        # Get the T latest clicked articles by the user
        old_clicks = history["article_id_fixed"]
        old_clicks = old_clicks[-min(len(old_clicks), self.T):]

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

class Collator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pos_inputs = self.tokenizer([p for p, _ in batch], return_tensors='pt', padding=True, truncation=True)
        neg_inputs = self.tokenizer([n for _, n in batch], return_tensors='pt', padding=True, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            pos_targets = self.tokenizer(['ja' for _ in batch], return_tensors="pt", padding=True, truncation=True)
            neg_targets = self.tokenizer(['nej' for _ in batch], return_tensors="pt", padding=True, truncation=True)
        
        empt = self.tokenizer(['<s>' for _ in batch], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "pos_input_ids": pos_inputs["input_ids"],
            "pos_attention_mask": pos_inputs["attention_mask"],
            "neg_input_ids": neg_inputs["input_ids"],
            "neg_attention_mask": neg_inputs["attention_mask"],
            "pos_labels": pos_targets["input_ids"],
            "neg_labels": neg_targets["input_ids"],
            "empt": empt["input_ids"]
        }

def get_loader(args, split, tokenizer, T=5, debug=False):
    """
    input:
        - args
        - split: str, one of 'train', 'validation', 'test'
        - T: int, number of previous clicked articles to consider
    """
    # test werkt nog niet
    assert args.dataset in ['demo', 'large'], 'dataset should be one of demo, large'
    assert split in ['train', 'validation', 'test'], 'dataset should be one of train, dev, test'

    collator = Collator(tokenizer)
    data = EkstraBladetDataset(create_prompt, dataset=args.dataset, split=split, T=T, debug=debug)

    return DataLoader(data, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers, shuffle=True)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    collator = Collator(tokenizer)
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