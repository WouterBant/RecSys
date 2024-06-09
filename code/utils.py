import torch


def compute_rank_loss(prob_pos, prob_neg):
    diff = torch.sigmoid(prob_pos - prob_neg)
    return -torch.log(1e-8 + diff)