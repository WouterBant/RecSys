import torch
from torch import nn
from transformers import MT5ForQuestionAnswering

from .base_model import BaseModel


class QA_fast_model(BaseModel):
    """
    Provide all condidate articles and assign probabilities to each.
    Cannot use rank loss anymore in this case.
    """

    def __init__(self, args):
        super(QA_fast_model, self).__init__(args)
        self.model = MT5ForQuestionAnswering.from_pretrained(args.backbone).to(self.device)
        self.ce = nn.CrossEntropyLoss()
    
    def train_step(self, batch):
        # Forward all examples simultaneously
        outputs = self.model(
            input_ids=batch["pos_input_ids"].to(self.device), 
            attention_mask=batch["pos_attention_mask"].to(self.device),
            decoder_input_ids=batch["decoder_start"].to(self.device)
        )
        logits = outputs.start_logits
        probs = torch.softmax(logits, dim=-1)
        loss_nll = self.ce(logits, batch["targets"].to(self.device))
        correct_prob_mean = probs[torch.arange(logits.size(0)),batch["targets"]].mean(dim=0).item()
    
        return loss_nll, correct_prob_mean, torch.zeros_like(correct_prob_mean)
    
    def validation_step(self, batch):
        outputs = self.validation_step_forward(batch)
        logits = outputs.start_logits
        probs = torch.softmax(logits, dim=-1)
        # TODO return the right probabilities and not all of them
        return probs