import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def train_step(self, batch):
        raise NotImplementedError

    def train_step_forward(self, batch):
        # Forward pass for the positive and negative examples
        pos_outputs = self.model(
            input_ids=batch["pos_input_ids"].to(self.device), 
            attention_mask=batch["pos_attention_mask"].to(self.device),
            decoder_input_ids=batch["decoder_start"].to(self.device),
        )
        neg_outputs = self.model(
            input_ids=batch["neg_input_ids"].to(self.device),
            attention_mask=batch["neg_attention_mask"].to(self.device),
            decoder_input_ids=batch["decoder_start"].to(self.device),
        )
        return pos_outputs, neg_outputs

    def validation_step(self, batch):
        raise NotImplementedError
    
    def validation_step_forward(self, batch):
        # Forward pass for the positive and negative examples together
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["prompt_input_ids"].to(self.device), 
                attention_mask=batch["prompt_attention_mask"].to(self.device),
                decoder_input_ids=batch["decoder_start"].to(self.device)
            )
        return outputs

    def test_step(self, batch):
        raise NotImplementedError
    
    def compute_rank_loss(self, prob_pos, prob_neg):
        diff = torch.sigmoid(prob_pos - prob_neg)
        return -torch.log(1e-8 + diff)