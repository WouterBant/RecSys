import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import MT5ForConditionalGeneration

from .base_model import BaseModel


class CG_classifier_model(BaseModel):
    """
    This class is a wrapper for the Conditional Generation mT5 model,
    but we put a classifier on top on the layer before the lm_head.
    """

    def __init__(self, args):
        super(CG_classifier_model, self).__init__(args)
        self.model = MT5ForConditionalGeneration.from_pretrained(args.backbone)
        classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.model.lm_head = classifier
        self.model.to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.ce = nn.BCELoss()

    def train_step(self, batch):
        pos_prob, neg_prob = self.train_step_forward(batch)
        pos_prob, neg_prob = pos_prob.logits[:,0,:], neg_prob.logits[:,0,:]  # these are now actually probabilities

        # Define target tensors
        pos_target = torch.ones(pos_prob.size(), device=self.device)
        neg_target = torch.zeros(neg_prob.size(), device=self.device)

        # Compute loss
        loss_nll = self.ce(pos_prob, pos_target) + self.ce(neg_prob, neg_target)
        loss_bpr = self.compute_rank_loss(pos_prob, neg_prob).mean(dim=0)
        loss = (1-self.args.labda)*loss_nll + self.args.labda*loss_bpr

        return loss, pos_prob, neg_prob
    
    def validation_step(self, batch):
        probs = self.validation_step_forward(batch).logits[:,0,:]
        return probs