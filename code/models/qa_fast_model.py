import torch
from torch import nn
from transformers import MT5ForQuestionAnswering

from .base_model import BaseModel


class QA_fast_model(BaseModel):
    """
    Provide all condidate articles and assign probabilities to each.
    Cannot use pairwise rank loss anymore in this case.
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
        assert logits.shape == batch["decoder_start"].shape

        # Compute the cross-entropy loss
        loss_nll = self.ce(logits, batch["targets_idxs"].to(self.device))

        # Get probabilities of the correct targets
        probs = torch.softmax(logits, dim=-1)
        correct_prob = probs[torch.arange(probs.size(0)), batch["targets_idxs"]]

        # Margin-based ranking loss
        # Create a mask to exclude the correct class probabilities from the ranking loss calculation
        mask = torch.ones_like(probs).scatter_(1, batch["targets_idxs"].unsqueeze(1).to(self.device), 0)
        incorrect_probs = probs * mask
        max_incorrect_prob = incorrect_probs.max(dim=-1).values

        # Ranking loss to ensure correct class probability is higher than incorrect by at least the margin
        rank_loss = torch.clamp(0.1 - (correct_prob - max_incorrect_prob), min=0).mean()

        # Combine the cross-entropy loss with the ranking loss
        combined_loss = loss_nll + self.args.labda*rank_loss

        return combined_loss, correct_prob.mean(), torch.tensor([0.0]).to(self.device)

    def validation_step(self, batch):
        # Forward all examples simultaneously
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["pos_input_ids"].to(self.device), 
                attention_mask=batch["pos_attention_mask"].to(self.device),
                decoder_input_ids=batch["decoder_start"].to(self.device)
            )
        logits = outputs.start_logits

        # NOTE the following works only with batch size 1
        logits = logits.squeeze()
        logits = logits[batch["decoder_start"].squeeze() == 1250]  # only keep seperator tokens
        probs = torch.softmax(logits, dim=-1)

        return probs