import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import MT5ForConditionalGeneration

from .base_model import BaseModel


class CG_model(BaseModel):
    """
    This class is a wrapper for the Conditional Generation mT5 model.
    This is to our understanding the closest model to the one used by PGNR.
    """

    def __init__(self, args):
        super(CG_model, self).__init__(args)
        self.model = MT5ForConditionalGeneration.from_pretrained(args.backbone)
        self.model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.ce = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def train_step(self, batch):
        pos_outputs, neg_outputs = self.train_step_forward(batch)

        # Only take the first token (should be 'ja' or 'nej')
        pos_logits = pos_outputs.logits[:, 0, :]  # B, T, V -> B, V
        neg_logits = neg_outputs.logits[:, 0, :]

        # 432 is the token id for 'ja'
        pos_prob_yes = torch.softmax(pos_logits, dim=-1)[:, 432]  # B, V -> B
        neg_prob_yes = torch.softmax(neg_logits, dim=-1)[:, 432]

        # Same for the targets that store one of the V labels
        pos_target = batch["pos_labels"][:, 0].to(self.device)  # B, T -> B
        neg_target = batch["neg_labels"][:, 0].to(self.device)

        # Compute loss
        loss_nll = self.ce(pos_logits, pos_target) + self.ce(neg_logits, neg_target)
        loss_bpr = self.compute_rank_loss(pos_prob_yes, neg_prob_yes).mean(dim=0)
        loss = (1 - self.args.labda) * loss_nll + self.args.labda * loss_bpr
        return loss, pos_prob_yes, neg_prob_yes

    def validation_step(self, batch):
        outputs = self.validation_step_forward(batch)

        # Only take the first token (should be 'ja' or 'nej')
        logits = outputs.logits[:, 0, :]  # B, T, V -> B, V

        # 36339 is the token id for 'ja'
        probs = torch.softmax(logits, dim=-1)[:, 432]  # B, V -> B
        return probs
