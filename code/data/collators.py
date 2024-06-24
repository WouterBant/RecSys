import torch
import warnings
warnings.filterwarnings("ignore")


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
        prompts = self.tokenizer([p for data in batch for p in data["prompts"]], return_tensors='pt', padding=True, truncation=True)
        
        targets = batch[0]["targets"]
        categories = batch[0]["categories"]
        publish_time = batch[0]["publish_time"]
        
        with self.tokenizer.as_target_tokenizer():
            decoder_start = self.tokenizer(['ja / nej' for data in batch for p in data["prompts"]], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "prompt_input_ids": prompts["input_ids"],
            "prompt_attention_mask": prompts["attention_mask"],
            "decoder_start": decoder_start["input_ids"],
            "targets": targets,
            "categories": categories,  # can be used for diversity evaluation
            "publish_time": publish_time
        }

class CollatorUnderstand:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # TODO now relies on batch size 1
        prompts = self.tokenizer([p for data in batch for p in data["prompts"]], return_tensors='pt', padding=True, truncation=True)
        
        targets = batch[0]["targets"]
        categories = batch[0]["categories"]
        
        with self.tokenizer.as_target_tokenizer():
            decoder_start = self.tokenizer(['ja / nej' for data in batch for p in data["prompts"]], return_tensors="pt", padding=True, truncation=True)
        
        return {
            "prompt_input_ids": prompts["input_ids"],
            "prompt_attention_mask": prompts["attention_mask"],
            "decoder_start": decoder_start["input_ids"],
            "targets": targets,
            "categories": categories,  # can be used for diversity evaluation
            "prompts": [p for data in batch for p in data["prompts"]]
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
            "targets": batch[0]["target_one_hot"],  # TODO assumes batch size 1 for eval
            "categories": batch[0]["categories"], 
        }