import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    log_loss,
)
from beyond_accuracy import (
    intralist_diversity,
    serendipity,
    coverage_count,
    coverage_fraction,
    novelty,
    index_of_dispersion
)


class MetricsEvaluator:
    def __init__(self, k=5):
        self.k = k

    def compute_metrics(self, output):
        scores = output['scores']
        labels = output['labels']
        # recommendations = output['recommendations']
        # candidate_items = output.get('candidate_items', [])
        # click_histories = output.get('click_histories', [])

        # top_k_recommendations = self.get_top_k_recommendations(recommendations, scores)
        
        return {
            f'ndcg@{self.k}': self.ndcg_at_k(scores, labels),
            f'mrr': self.mrr_at_k(scores, labels, 10**6),
            f'mrr@{self.k}': self.mrr_at_k(scores, labels, self.k),
            f'precision@{self.k}': self.precision_at_k(scores, labels),
            f'recall@{self.k}': self.recall_at_k(scores, labels),
            f'mean_squared_error@{self.k}': self.mean_squared_error_at_k(labels, scores),
            f'accuracy@{self.k}': self.accuracy_score_at_k(labels, scores),
            f'f1@{self.k}': self.f1_score_at_k(labels, scores),
            # 'log_loss': self.log_loss_at_k(labels, scores),
            # 'intralist_diversity': intralist_diversity(top_k_recommendations),
            # 'coverage_count': coverage_count(recommendations),
            # 'coverage_fraction': coverage_fraction(recommendations, candidate_items),
            # 'serendipity': serendipity(recommendations, click_histories),
            # 'novelty': novelty(recommendations),
            # 'index_of_dispersion': index_of_dispersion(recommendations.flatten())
        }

    def get_top_k_recommendations(self, recommendations, scores):
        top_k_recommendations = []
        for rec_list, score_list in zip(recommendations, scores):
            top_k_indices = np.argsort(score_list)[::-1][:self.k]
            top_k_recommendations.append([rec_list[i] for i in top_k_indices])
        return np.array(top_k_recommendations)

    def ndcg_at_k(self, scores, labels):
        best = self.dcg_score_at_k(labels, labels)
        actual = self.dcg_score_at_k(labels, scores)
        return actual / best

    def mrr_at_k(self, scores, labels, k):
        order = np.argsort(scores)[::-1]
        k = min(k, len(order))
        labels = np.take(labels, order[:k])
        denom = np.sum(labels)
        if denom == 0:
            return 0.0
        rr_score = labels / (np.arange(len(labels)) + 1)
        return np.sum(rr_score) / denom

    def dcg_score_at_k(self, scores, labels):
        k = min(len(labels), self.k)
        order = np.argsort(scores)[::-1]
        y_true = np.take(labels, order[:k])
        gains = 2**y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def mean_squared_error_at_k(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1][:self.k]
        return mean_squared_error(np.take(y_true, order), np.take(y_pred, order))

    def accuracy_score_at_k(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1][:self.k]
        y_pred_at_k = np.take(y_pred, order)
        y_true_at_k = np.take(y_true, order)
        y_pred_at_k_binary = (y_pred_at_k > 0.5).astype(int)
        return accuracy_score(y_true_at_k.flatten(), y_pred_at_k_binary.flatten())

    def f1_score_at_k(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1][:self.k]
        y_pred_at_k = np.take(y_pred, order)
        y_true_at_k = np.take(y_true, order)
        y_pred_at_k_binary = (y_pred_at_k > 0.5).astype(int)
        return f1_score(y_true_at_k.flatten(), y_pred_at_k_binary.flatten())

    def log_loss_at_k(self, y_true, y_pred):
        order = np.argsort(y_pred)[::-1][:self.k]
        return log_loss(np.take(y_true, order), np.take(y_pred, order))

    def precision_at_k(self, scores, labels):
        order = np.argsort(scores)[::-1]
        labels = np.take(labels, order[:self.k])
        return np.sum(labels) / self.k

    def recall_at_k(self, scores, labels):
        order = np.argsort(scores)[::-1]
        labels = np.take(labels, order[:self.k])
        denom = np.sum(labels[:self.k])
        if denom == 0:
            return 0
        return np.sum(labels) / denom