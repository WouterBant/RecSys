import torch
from torch import nn
from transformers import MT5ForQuestionAnswering

from .base_model import BaseModel


class QA_model(BaseModel):
    """
    This class is a wrapper for the Question Answering mT5 model.
    """

    def __init__(self, args):
        super(QA_model, self).__init__(args)
        self.model = MT5ForQuestionAnswering.from_pretrained(args.backbone)
        self.model.to(self.device)
        self.ce = nn.CrossEntropyLoss()

    def train_step(self, batch):
        pos_outputs, neg_outputs = self.train_step_forward(batch)

        # Get the start logits
        pos_logits = pos_outputs.start_logits
        neg_logits = neg_outputs.start_logits

        # Compute the probabilities
        pos_probs = torch.softmax(pos_logits, dim=-1)
        neg_probs = torch.softmax(neg_logits, dim=-1)
        pos_prob_yes = pos_probs[:, 0]  # B,T -> B
        neg_prob_yes = neg_probs[:, 0]  # B,T -> B

        # Create the targets
        batch_size = pos_probs.shape[0]

        # 0 = idx of 'ja' and 1 = idx of 'nej' in decoder input sequence
        pos_target = torch.tensor(batch_size * [0]).to(self.device)
        neg_target = torch.tensor(batch_size * [3]).to(self.device)

        # Compute the loss
        loss_nll = self.ce(pos_logits, pos_target) + self.ce(neg_logits, neg_target)
        loss_bpr = self.compute_rank_loss(pos_prob_yes, neg_prob_yes).mean(dim=0)
        loss = (1 - self.args.labda) * loss_nll + self.args.labda * loss_bpr

        return loss, pos_prob_yes, neg_prob_yes

    def validation_step(self, batch):
        outputs = self.validation_step_forward(batch)

        # Only consider the start logits
        logits = outputs.start_logits

        # Only consider the probability for 'ja'
        probs = torch.softmax(logits, dim=-1)[:, 0]
        return probs
