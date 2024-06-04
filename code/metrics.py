import torch


def compute_metrics(output, k=10):
    scores, labels = output['scores'], output['labels']
    return {
        'ndcg': ndcg_at_k(k, scores, labels),
        'mrr': mrr_at_k(k, scores, labels),
        'precision': precision_at_k(k, scores, labels),
        'recall': recall_at_k(k, scores, labels),
    }

def ndcg_at_k(k, scores, labels):
    pass

def mrr_at_k(k, scores, labels):
    pass

def precision_at_k(k, scores, labels):
    pass

def recall_at_k(k, scores, labels):
    pass
