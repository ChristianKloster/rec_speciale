from operator import itemgetter
from typing import List, Tuple

import numpy as np


class Metrics2():
    def __init__(self, predictions: np.ndarray, actuals: np.ndarray, k: int, metrics: str = ''):
        """
        Constructor for Metrics2

        :param predictions: n x latent_factors numpy array - n being number of users
        :param actuals: m x latent_factors numpy array - m being number of items
        :param k: length of recommended list
        :param metrics: which metrics to compute (default is all) separated by comma. Can be mae, rmse, precision, recall, ndcg and mrr
        """
        self.predictions = predictions
        self.actuals = actuals
        self.k = k
        self.metrics = metrics
        pass

    def calculate(self):
        """
        Calculates metrics
        """
        metrics = self.metrics.replace(' ', '').split(',')
        valid_metrics = ['', 'rmse', 'mae', 'ndcg', 'precision', 'recall', 'mrr', 'hr']
        assert all(m in valid_metrics for m in metrics), 'Invalid metrics!'

        res_dict = {}
        n_users, _ = self.predictions.shape


        # computing metrics
        precision = 0
        recall = 0
        ndcg = 0
        mrr = 0
        hr = 0
        for u in range(n_users):
            acts = self.actuals[u]
            preds = self.predictions[u]
            ratings = self.__build_pred_actual_tuples(preds, acts)
            avg_rating = float(np.mean(acts[acts > 0]))

            # precision and recall
            if 'precision' in metrics or 'recall' in metrics or '' in metrics:
                user_prec, user_recall = self.__precision_and_recall(ratings, self.k, avg_rating)
                precision += user_prec
                recall += user_recall
            # ndcg
            if 'ndcg' in metrics or '' in metrics:
                user_ndcg = self.__ndcg(ratings, acts, self.k)
                ndcg += user_ndcg
            # mrr
            if 'mrr' in metrics or '' in metrics:
                user_mrr = self.__mrr(ratings, avg_rating)
                mrr += user_mrr
            if 'hr' in metrics or '' in metrics:
                user_hr = self.__hr(ratings, acts, self.k)
                hr += user_hr


        # averaging metrics
        # precision and recall
        if 'precision' in metrics or 'recall' in metrics or '' in metrics:
            res_dict['precision'] = precision / n_users
            res_dict['recall'] = recall / n_users
        if 'ndcg' in metrics or '' in metrics:
            res_dict['ndcg'] = ndcg / n_users
        if 'mrr' in metrics or '' in metrics:
            res_dict['mrr'] = mrr / n_users
        if 'rmse' in metrics or '' in metrics:
            res_dict['rmse'] = self.__rmse()
        if 'mae' in metrics or '' in metrics:
            res_dict['mae'] = self.__mae()
        if 'hr' in metrics or '' in metrics:
            res_dict['hr'] = hr / n_users

        return res_dict

    def __build_pred_actual_tuples(self, predictions: np.ndarray, actuals: np.ndarray):
        # generating (prediction, actual) tuples
        ratings = [(pred, act) for (pred, act) in zip(predictions, actuals)]
        # sorting based on predictions
        ratings.sort(reverse=True, key=itemgetter(0))
        return ratings

    def __ndcg(self, ratings: List[Tuple[float, int]], actuals: np.ndarray, k: int):
        """
        Returns the normalized discounted cumulative gain(nDCG) at **k** recommendations
        where order of most relevant items matters, most relevant first ect..
        :param ratings: list of tuples (prediction, actual) sorted descending order
        :param k: k is a constant that tells how long the recommendation list is
        """
        n_rated = sum(actuals > 0)
        k = min(k, n_rated)

        dcg_relevances = [act for (_, act) in ratings[:k]]
        sorted_actuals = np.sort(actuals)[::-1]
        idcg_relevances = [act for act in sorted_actuals[:k]]

        dcg = np.sum((np.power(2, dcg_relevances) - 1) / np.log2(np.arange(2, len(dcg_relevances) + 2)))
        idcg = np.sum((np.power(2, idcg_relevances) - 1) / np.log2(np.arange(2, len(idcg_relevances) + 2)))

        return dcg / idcg

    def __precision_and_recall(self, ratings: List[Tuple[float, int]], k: int, avg_rating: float):
        """
        Returns the precision and recall in the top k recommended list to a specific user
        :param avg_rating: value to determine if a person likes or dislikes an item depended on
            the users avg rating of all items rated
        :param ratings: list of tuples (prediction, actual) sorted descending order
        :param k: k is a constant that tells how long the recommendation list is
        """
        n_rated = len([act for (_, act) in ratings if act > 0])
        # Number of relevant items
        n_rel = sum((act >= 4) for (_, act) in ratings)
        # Number of relevant items which are recommended
        n_rel_and_rec = sum((act >= 4) for (_, act) in ratings[:k])

        # Fraction of recommended items that are relevant
        precision = n_rel_and_rec / k
        # Fraction of relevant items that are recommended
        recall = n_rel_and_rec / n_rel if n_rel > 0 else 1

        return precision, recall

    def __rmse(self):
        """
        Computes root mean squared error
        """
        return np.sqrt(np.mean((self.predictions[self.actuals > 0] - self.actuals[self.actuals > 0]) ** 2))

    def __mae(self):
        """
        Computes mean absolute error
        """
        return np.mean(abs(self.predictions[self.actuals > 0] - self.actuals[self.actuals > 0]))

    def __mrr(self, ratings: List[Tuple[float, int]], avg_rating: float):
        """
        returns mean reciprocal rank
        :param ratings: list of tuples (prediction, actual) sorted descending order
        """
        binary_relevance = [1 if act >= avg_rating else 0
                            for (_, act) in ratings]
        n_relevants = sum(binary_relevance)

        return 1 / (binary_relevance.index(1) + 1) if n_relevants != 0 else 0

    @staticmethod
    def __hr(ratings: List[Tuple[float, int]], actuals: np.ndarray, k: int):
        """
        Computes the hit-rate of each
        :param k: number of items recommended
        :param ratings: list of tuples (prediction, actual) sorted descending order
        """
        n_rated = sum(actuals > 0)
        n_hits = len([act for (_, act) in ratings[:k] if act > 0])
        return n_hits/min(k, n_rated)





if __name__ == '__main__':
    #    actuals = np.random.randint(6, size=1700)
    #    preds = np.random.randint(1, 6, size=1700)
    actuals = np.array([[0, 2, 5]])
    preds = np.array([[5, 4, 5]])
    n = Metrics2(preds, actuals, 2, metrics='hr')
    print(n.calculate())

#   x = build_pred_actual_tuples(preds, actuals)
#    prec, reca = precision_and_recall(preds, actuals, 5)
#    normalizedcg = ndcg(preds, actuals, 5)
#    trmse = rmse(preds, actuals)
#    tmae = mae(preds, actuals)
#    print(tmae)
#    print(trmse)
#    print(normalizedcg)
#    print(prec)
#    print(reca)
