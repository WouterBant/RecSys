import torch
import numpy as np

try:
    from sklearn.metrics import (
        # _regression:
        mean_squared_error,
        # _ranking:
        roc_auc_score,
        # _classification:
        accuracy_score,
        f1_score,
        log_loss,
    )
except ImportError:
    print("sklearn not available")

def compute_metrics(output, k=10):
    scores, labels = output['scores'], output['labels']
    return {
        'ndcg': ndcg_at_k(k, scores, labels),
        'mrr': mrr_at_k(k, scores, labels),
        'precision': precision_at_k(k, scores, labels),
        'recall': recall_at_k(k, scores, labels),
        'mean_squared_error': mean_squared_error_at_k(k, labels, scores),
        'roc_auc': roc_auc_score_at_k(k, labels, scores),
        'accuracy': accuracy_score_at_k(k, labels, scores),
        'f1': f1_score_at_k(k, labels, scores),
        'log_loss': log_loss_at_k(k, labels, scores),
    }

def ndcg_at_k(k, scores, labels):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) score at a rank `k`.

    Args:
        y_true (np.ndarray): A 1D or 2D array of ground-truth relevance labels.
                            Each element should be a non-negative integer. In case
                            of a 2D array, each row represents a different sample.
        y_pred (np.ndarray): A 1D or 2D array of predicted scores. Each element is
                            a score corresponding to the predicted relevance. The
                            array should have the same shape as `y_true`.
        k (int, optional): The rank at which the NDCG score is calculated. Defaults
                            to 10. If `k` is larger than the number of elements, it
                            will be truncated to the number of elements.

    Returns:
        float: The calculated NDCG score for the top `k` elements. The score ranges
                from 0 to 1, with 1 representing the perfect ranking.

    Examples:
        >>> labels = np.array([1, 0, 0, 1, 0])
        >>> scores = np.array([0.1, 0.2, 0.1, 0.8, 0.4])
        >>> ndcg_score(labels, scores)
            0.5249810332008933
    """
    best = dcg_score_at_k(k, labels, labels)
    actual = dcg_score_at_k(k, labels, scores)
    return actual / best

def mrr_at_k(k, scores, labels):
    """Computes the Mean Reciprocal Rank (MRR) score at rank k.

    Args:
        k (int): The rank position up to which the MRR is computed.
        scores (np.ndarray): A 1D array of predicted scores. These scores indicate the likelihood
                             of items being relevant.
        labels (np.ndarray): A 1D array of ground-truth labels. These should be binary (0 or 1),
                             where 1 indicates the relevant item.

    Returns:
        float: The mean reciprocal rank (MRR) score at rank k.

    Note:
        Both `scores` and `labels` should be 1D arrays of the same length.
        The function assumes higher scores in `scores` indicate higher relevance.

    Examples:
        >>> labels = np.array([1, 0, 0, 1, 0])
        >>> scores = np.array([0.5, 0.2, 0.1, 0.8, 0.4])
        >>> mrr_at_k(3, scores, labels)
            0.5
    """
    order = np.argsort(scores)[::-1]
    labels = np.take(labels, order[:k])
    rr_score = labels / (np.arange(len(labels)) + 1)
    return np.sum(rr_score) / np.sum(labels)

def dcg_score_at_k(k, scores, labels):
    """
    Compute the Discounted Cumulative Gain (DCG) score at a particular rank `k`.

    Args:
        labels (np.ndarray): A 1D or 2D array of ground-truth relevance labels.
                             Each element should be a non-negative integer.
        scores (np.ndarray): A 1D or 2D array of predicted scores. Each element is
                             a score corresponding to the predicted relevance.
        k (int): The rank at which the DCG score is calculated. If `k` is larger
                 than the number of elements, it will be truncated to the number
                 of elements.

    Note:
        In case of a 2D array, each row represents a different sample.

    Returns:
        float or np.ndarray: The calculated DCG score for the top `k` elements.
                             If the input is 2D, an array of DCG scores is returned.

    Raises:
        ValueError: If `labels` and `scores` have different shapes.

    Examples:
        >>> labels = np.array([1, 0, 0, 1, 0])
        >>> scores = np.array([0.5, 0.2, 0.1, 0.8, 0.4])
        >>> dcg_score_at_k(3, scores, labels)
        1.63
    """
    k = min(len(labels), k)
    order = np.argsort(scores)[::-1]
    y_true = np.take(labels, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def mean_squared_error_at_k(k, y_true, y_pred):
    order = np.argsort(y_pred)[::-1][:k]
    return mean_squared_error(np.take(y_true, order), np.take(y_pred, order))

def roc_auc_score_at_k(k, y_true, y_pred):
    order = np.argsort(y_pred)[::-1][:k]
    return roc_auc_score(np.take(y_true, order), np.take(y_pred, order))

def accuracy_score_at_k(k, y_true, y_pred):
    order = np.argsort(y_pred)[::-1][:k]
    y_pred_at_k = np.take(y_pred, order)
    y_true_at_k = np.take(y_true, order)
    y_pred_at_k_binary = (y_pred_at_k > 0.5).astype(int)
    return accuracy_score(y_true_at_k, y_pred_at_k_binary)

def f1_score_at_k(k, y_true, y_pred):
    order = np.argsort(y_pred)[::-1][:k]
    y_pred_at_k = np.take(y_pred, order)
    y_true_at_k = np.take(y_true, order)
    y_pred_at_k_binary = (y_pred_at_k > 0.5).astype(int)
    return f1_score(y_true_at_k, y_pred_at_k_binary)

def log_loss_at_k(k, y_true, y_pred):
    order = np.argsort(y_pred)[::-1][:k]
    return log_loss(np.take(y_true, order), np.take(y_pred, order))

def precision_at_k(k, scores, labels):
    order = np.argsort(scores)[::-1]
    labels = np.take(labels, order[:k])
    return np.sum(labels) / k

def recall_at_k(k, scores, labels):
    order = np.argsort(scores)[::-1]
    labels = np.take(labels, order[:k])
    return np.sum(labels) / np.sum(labels[:k])
