import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random
import polars as pl
from transformers import AutoTokenizer


class EkstraBladetDataset(Dataset):

    def __init__(self, create_prompt, tokenizer, split="train", T=5):
        
        # Download the dataset from huggingface
        self.behaviors = load_dataset('Wouter01/RecSys_demo', 'behaviors', cache_dir="demo_data")[split]
        self.articles = load_dataset('Wouter01/RecSys_demo', 'articles', cache_dir="demo_data")[split].to_pandas()
        self.history = load_dataset('Wouter01/RecSys_demo', 'history', cache_dir="demo_data")[split].to_pandas()

        # Set fast lookup for identifier keys
        self.history.set_index("user_id", inplace=True)
        self.articles.set_index("article_id", inplace=True)

        self.T = T  # Number of previous clicked articles to consider
        self.create_prompt = create_prompt  # Function to create a prompt from the data
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        # Every item consits of a positive and negative sample
        behavior = self.behaviors[idx]

        # Pick random positive and negative samples
        clicked_articles = behavior["article_ids_clicked"]
        unclicked_articles = [article for article in behavior["article_ids_inview"] if article not in clicked_articles]
        pos_sample = random.choice(clicked_articles)
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

        return (self.tokenize(pos_prompt, padding='max_length', max_length=4096, truncation=True, return_tensors='pt'), 
                self.tokenize(neg_prompt, padding='max_length', max_length=4096, truncation=True, return_tensors='pt'))
    
def create_prompt(titles, subtitles, title, subtitle):
    # TODO maak vet mooie deense prompt hier
    prompt = f"Given the following titles and subtitles of previously read articles:\n"
    for i, (t, s) in enumerate(zip(titles, subtitles)):
        prompt += f"Article {i+1}:\nTitle: {t}\nSubtitle: {s}\n\n"
    prompt += f"Is the user likely to click on an articles with title {title} and subtitle {subtitle}? (yes/no)\n"
    return prompt

def collate_fn(batch):
    tokenized_pos = [prompt[0] for prompt in batch]
    tokenized_neg = [prompt[1] for prompt in batch]

    pos_input_ids = torch.cat([item['input_ids'] for item in tokenized_pos], dim=0)
    pos_attention_mask = torch.cat([item['attention_mask'] for item in tokenized_pos], dim=0)

    neg_input_ids = torch.cat([item['input_ids'] for item in tokenized_neg], dim=0)
    neg_attention_mask = torch.cat([item['attention_mask'] for item in tokenized_neg], dim=0)

    return {
        'pos_input_ids': pos_input_ids,
        'pos_attention_mask': pos_attention_mask,
        'neg_input_ids': neg_input_ids,
        'neg_attention_mask': neg_attention_mask
    }

def get_loader(dataset, T=5):
    """
    input:
        - dataset: str, one of 'train', 'validation', 'test'
        - T: int, number of previous clicked articles to consider
    """
    # TODO geef tokenizer mee hier
    # test werkt nog niet
    assert dataset in ['train', 'validation', 'test'], 'dataset should be one of train, dev, test'

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", model_max_length=4096)
    data = EkstraBladetDataset(create_prompt, tokenizer, split=dataset, T=T)

    return DataLoader(data, batch_size=64, collate_fn=collate_fn, shuffle=True)