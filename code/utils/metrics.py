import numpy as np


class MetricsEvaluator:
    def __init__(self, k=5, T=4):
        self.k = k
        self.T = T

    def compute_metrics(self, output):
        scores = output['scores']
        labels = output['labels']
        categories = output['categories']

        assert scores.shape == labels.shape, f"{scores.shape} {labels.shape}"
        
        return {
            f'AUC': self.auc(scores, labels),
            f'ndcg': self.ndcg_at_k(scores, labels, 10**6),
            f'ndcg@{self.k}': self.ndcg_at_k(scores, labels, self.k),
            f'mrr': self.mrr_at_k(scores, labels, 10**6),
            f'mrr@{self.k}': self.mrr_at_k(scores, labels, self.k),
            f'precision@1': self.precision_at_k(scores, labels, 1),
            f'recall@{self.k}': self.recall_at_k(scores, labels),
            f'hit_ratio@{self.k}': self.hit_ratio_at_k(scores, labels),
            f'diversity@1': self.diversity_at_k(scores, categories, k=1),
            f'diversity@2': self.diversity_at_k(scores, categories, k=2),
            f'diversity@3': self.diversity_at_k(scores, categories, k=3),
            f'diversity@4': self.diversity_at_k(scores, categories, k=4),
            f'diversity@5': self.diversity_at_k(scores, categories, k=5),
            f'intra_list_diversity@1': self.intra_list_diversity_at_k(scores, categories, k=1),  # just a check
            f'intra_list_diversity@2': self.intra_list_diversity_at_k(scores, categories, k=2),
            f'intra_list_diversity@3': self.intra_list_diversity_at_k(scores, categories, k=3),
            f'intra_list_diversity@4': self.intra_list_diversity_at_k(scores, categories, k=4),
            f'intra_list_diversity@5': self.intra_list_diversity_at_k(scores, categories, k=5),
        }

    def get_top_k_recommendations(self, recommendations, scores):
        top_k_recommendations = []
        for rec_list, score_list in zip(recommendations, scores):
            top_k_indices = np.argsort(score_list)[::-1][:self.k]
            top_k_recommendations.append([rec_list[i] for i in top_k_indices])
        return np.array(top_k_recommendations)

    def ndcg_at_k(self, scores, labels, k=5):
        best = self.dcg_score_at_k(labels, labels, k)
        actual = self.dcg_score_at_k(scores, labels, k)
        return actual / best

    def dcg_score_at_k(self, scores, labels, k):
        k = min(len(labels), k)
        order = np.argsort(scores)[::-1]
        y_true = np.take(labels, order[:k])
        gains = 2**y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def mrr_at_k(self, scores, labels, k):
        order = np.argsort(scores)[::-1]
        k = min(k, len(order))
        labels = np.take(labels, order[:k])
        rr_score = 0.0
        for i in range(k):
            if labels[i] == 1:
                rr_score = 1.0 / (i + 1)
                break
        return rr_score

    def precision_at_k(self, scores, labels, k=1):
        order = np.argsort(scores)[::-1]
        labels = np.take(labels, order[:k])
        return np.sum(labels) / k

    def recall_at_k(self, scores, labels):
        order = np.argsort(scores)[::-1][:self.k]
        labels_at_k = np.take(labels, order)
        return np.sum(labels_at_k) / np.sum(labels) if np.sum(labels) != 0 else 0

    def hit_ratio_at_k(self, scores, labels):
        order = np.argsort(scores)[::-1][:self.k]
        labels_at_k = np.take(labels, order)
        return np.any(labels_at_k)

    def diversity_at_k(self, scores, categories, k):
        # Count the number of categories in topk that were not in the clicked categories
        clicked_categories = categories[:self.T]
        inview_categories = categories[self.T:]
        order = np.argsort(scores)[::-1][:k]
        inview_categories_at_k = np.take(inview_categories, order)
        diversity = 0
        for category in inview_categories_at_k:
            if category not in clicked_categories:
                diversity += 1
        return diversity / k

    def intra_list_diversity_at_k(self, scores, categories, k):
        # Count the number of categories in topk that are unique
        inview_categories = categories[self.T:]
        order = np.argsort(scores)[::-1][:k]
        inview_categories_at_k = np.take(inview_categories, order)
        return len(set(inview_categories_at_k)) / k
    
    def auc(self, scores, labels):
        order = np.argsort(scores)[::-1]
        labels = np.take(labels, order)
        rr_score = 0.0
        for i in range(len(scores)):
            if labels[i] == 1:
                return (len(scores) - i)/len(scores)
        return 0
