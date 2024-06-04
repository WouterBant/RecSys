import torch
import torch.nn as nn
import numpy as np
from modeling_p5 import P5


class P5Pretraining(P5):
    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.losses.split(',')

        # WB: TODO - make this a hyperparameter
        # btw this was the best setting so for default fine
        self.weight = 0.40

    def train_step(self, batch):

        device = next(self.parameters()).device

        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)
        lm_labels = batch["target_ids"].to(device)

        # WB: loss_weights is 1 / length of target text
        loss_weights = batch["loss_weights"].to(device)

        # WB: Here is the forward pass call
        output = self(
            input_ids = input_ids,
            whole_word_ids = whole_word_ids,
            labels = lm_labels,
            return_dict = True
        )

        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size() # B = len(batch) * 2

        # predicted 'word' loss
        loss = output['loss'] 
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1) # for each sample, compute its average loss

        assert 'logits' in output
        
        # WB: Here is the pair loss calculation
        target_text = batch['target_text']
        yes_mask = np.array(target_text) == 'yes'  # true/false
        yes_mask = torch.from_numpy(yes_mask)[:B // 2].to(device)  
        no_mask = np.array(target_text) == 'no'  # true/false
        no_mask = torch.from_numpy(no_mask)[:B // 2].to(device)  

        SOFTMAX = nn.Softmax(dim = -1)
        logits = output['logits'].to(device) #[batch, time, vocab_size]
        prob = SOFTMAX(logits) #[batch, time, vocab_size]

        # WB: FIXME - hard coded token id
        pos_prob = prob[:, 0, :][:, 4273] # tensor with size B
        first = pos_prob[: B//2]
        second = pos_prob[B//2:]

        # WB: is this not simply pos_diff*2?
        pos_diff = (first - second) * yes_mask
        neg_diff = (first - second) * no_mask * -1
        diff = pos_diff + neg_diff # pos - neg for all half
        
        # WB: diff is the difference between the two probabilities
        pair_loss = -(diff).sigmoid().log().sum()
        average_pair_loss = pair_loss/(B//2)

        results = {}

        # loss: average loss for each sample, all loss \in [0, 1]
        results['loss'] = (1 - self.weight)*(loss * loss_weights).mean()  + self.weight * average_pair_loss

        results['pair_loss'] = pair_loss.detach()
        results['pair_loss_count'] = len(loss)//2

        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device

        input_ids = batch['input_ids'].to(device)
        lm_labels = batch["target_ids"].to(device)
        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids =input_ids,
            labels=lm_labels,
            return_dict=True
        )

        assert 'loss' in output

        # WB: -100 is the padding token
        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()  # B = len(batch)

        # predicted 'word' loss
        loss = output['loss']  # cross-entropy loss for all samples len(batch)
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # for each sample, compute its average loss

        results = {}
        results['loss'] = (loss * loss_weights).mean()  # minimize this loss
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss # average loss for each sample
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    
    def force_one_paragraph(self, batch_id, previous_token_ids):
        previous_token_ids = previous_token_ids.tolist()
        if 4273 in previous_token_ids or 150 in previous_token_ids:
            return [1] # eos token id
        return [150, 4273]

    @torch.no_grad()
    def generate_step(self, batch):
        self.eval()
        device = next(self.parameters()).device

        user_id = batch['user_id']
        item_id = batch['item_id']
        impress_id = batch['impress_id']
        target_text = batch['target_text']

        beam_outputs = self.generate(
                batch['input_ids'].to(device), 
                prefix_allowed_tokens_fn = self.force_one_paragraph,
                max_length=3, 
                num_beams=1,
                no_repeat_ngram_size=0, 
                num_return_sequences=1,
                early_stopping=True,
                output_scores = True,
            return_dict_in_generate= True)
    
        generated_sents = self.tokenizer.batch_decode(beam_outputs['sequences'], skip_special_tokens=True)
        prob = torch.softmax(beam_outputs['scores'][0], dim = -1)   # for the first token
        prob_yes = prob[:, 4273].tolist()  # WB: FIXME - hard coded token id for 'yes'?

        return user_id, impress_id, item_id, target_text, prob_yes, generated_sents