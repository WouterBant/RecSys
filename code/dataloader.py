from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from prompt_templates import *
from collators import *


class EkstraBladetDataset(Dataset):

    def __init__(self, args, create_prompt, split="train", debug=False):
        self.split = split
        
        # Download the dataset from huggingface
        if split == "test":
            self.behaviors = load_dataset(f'Wouter01/testbehaviors', cache_dir=f"../../testbehaviors_data")["train"]
            self.articles = load_dataset(f'Wouter01/testarticles', cache_dir=f"../../testarticles_data")["train"].to_pandas()
            self.history = load_dataset(f'Wouter01/testhistory', cache_dir=f"../../testhistory_data")["train"].to_pandas()
        elif args.evaltrain:  # TODO fix this also in understand notebook, seperate args for return type and data to use
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
        if args.dataset == "large" and split == "validation":  # TODO bit hacky
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
            titles_clicked, categories = [], []
            for article_id in old_clicks.tolist():
                article = self.articles.loc[article_id]
                titles_clicked.append(article["title"])
                categories.append(article["category_str"])
            
            random.shuffle(inview_articles)
            titles_inview = []
            for article_id in inview_articles:
                article = self.articles.loc[article_id]
                titles_inview.append(article["title"])
                categories.append(article["category_str"])
            
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
                "decoder_input": " > @ ".join(titles_inview) + " > @ " ,
                "target": target_idx,
                "target_one_hot": targets,
                "categories": categories,
            }

        if self.split == "validation" or self.split ==  "hoe":
            # Get the past article information
            titles, subtitles, categories, pubtimes = [], [], [], []
            for article_id in old_clicks.tolist():
                article = self.articles.loc[article_id]
                titles.append(article["title"])
                subtitles.append(article["subtitle"])
                categories.append(article["category_str"])
                pubtimes.append(article["published_time"])
            
            prompts = []
            for article_id in inview_articles:
                article = self.articles.loc[article_id]
                prompts.append(self.create_prompt({
                    "titles": titles,
                    "subtitles": subtitles,
                    "categories": categories[:self.args.T],
                    "publish_times": pubtimes,
                    "title": article["title"],
                    "subtitle": article["subtitle"],
                    "category": article["category_str"],
                    "publish_time": article["published_time"]
                }))
                categories.append(article["category_str"])
            
            clicked_articles = behavior["article_ids_clicked"]
            targets = [1 if article in clicked_articles else 0 for article in inview_articles]
            return {
                "prompts": prompts,
                "targets": targets,
                "categories": categories,
            }

        if self.split == "test":  # TODO only works for standard models
            # Get the past article information
            titles, subtitles, categories, pubtimes = [], [], [], []
            for article_id in old_clicks.tolist():
                article = self.articles.loc[article_id]
                titles.append(article["title"])
                subtitles.append(article["subtitle"])
                categories.append(article["category_str"])
                pubtimes.append(article["published_time"])
            
            prompts = []
            for article_id in inview_articles:
                article = self.articles.loc[article_id]
                
                prompts.append(self.create_prompt({
                    "titles": titles,
                    "subtitles": subtitles,
                    "categories": categories,
                    "publish_times": pubtimes,
                    "title": article["title"],
                    "subtitle": article["subtitle"],
                    "category": article["category_str"],
                    "publish_time": article["published_time"]
                }))
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
        titles, subtitles, categories, pubtimes = [], [], [], [] # last two are pos and neg samples
        for article_id in old_clicks.tolist() + [pos_sample, neg_sample]:
            article = self.articles.loc[article_id]
            titles.append(article["title"])
            subtitles.append(article["subtitle"])
            categories.append(article["category_str"])
            pubtimes.append(article["published_time"])

        assert len(titles) == self.T + 2 and len(titles) == len(subtitles) and len(titles) == len(categories) and len(titles) == len(pubtimes)
        
        title_pos, title_neg = titles[-2], titles[-1]
        subtitle_pos, subtitle_neg = subtitles[-2], subtitles[-1]
        category_pos, category_neg = categories[-2], categories[-1]
        pubtime_pos, pubtime_neg = pubtimes[-2], pubtimes[-1]
        titles, subtitles, categories, pubtimes = titles[:-2], subtitles[:-2], categories[:-2], pubtimes[:-2]

        # Create the prompts
        pos_prompt = self.create_prompt({
            "titles": titles,
            "subtitles": subtitles,
            "categories": categories,
            "publish_times": pubtimes,
            "title": title_pos,
            "subtitle": subtitle_pos,
            "category": category_pos,
            "publish_time": pubtime_pos
        })
        neg_prompt = self.create_prompt({
            "titles": titles,
            "subtitles": subtitles,
            "categories": categories,
            "publish_times": pubtimes,
            "title": title_neg,
            "subtitle": subtitle_neg,
            "category": category_neg,
            "publish_time": pubtime_neg
        })

        return pos_prompt, neg_prompt


def get_loader(args, split, tokenizer, debug=False):
    assert split in ['train', 'validation', 'test'], 'dataset should be one of train, dev, test'

    if args.model == "QA+":
        create_prompt = create_prompt_qa_fast
        collator = CollatorQAfast(tokenizer)
        data = EkstraBladetDataset(args, create_prompt, split=split)  # todo change this to split
        shuffle = (split == "train")
        bs = args.batch_size if split == "train" else 1
        return DataLoader(data, batch_size=bs, collate_fn=collator, num_workers=args.num_workers, shuffle=shuffle)

    if args.prompt == "titles":
        create_prompt = create_prompt_titles
    elif args.prompt == "subtitles":
        create_prompt = create_prompt_subtitles
    elif args.prompt == "diversity":
        create_prompt = create_prompt_diversity
    elif args.prompt == "pubtime":
        create_prompt = create_prompt_w_publishtime
    else:
        assert True == False # TODO fix this

    if split == "train":
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
    
    assert True == False, "python is broken"  # TODO fix this