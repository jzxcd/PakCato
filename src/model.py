from typing import List
from sklearn.cluster import OPTICS
import numpy as np


class Model:

    def _outlier_consolidate(self, cluster_ids: List[int]) -> List[int]:
        """
        To assign each outlier to it's own group id
        e.g. 
        cluster_ids = [-1, 1, 1, 2, -1, 3, 3] 
        output      = [ 1, 2, 2, 3,  4, 5, 5]
        """

        add = 0
        curated = []
        for i in range(len(cluster_ids)):

            if cluster_ids[i] != -1:
                curated.append(cluster_ids[i] + add)
            if cluster_ids[i] == -1:
                curated.append(curated[-1] + 1 if curated else 1)
                add += 1
        return curated


    def algo_grouping(self, scores: List[float]):
        for prev, curr in zip(scores, scores[1:]):
            assert prev >= curr, "scores input to be descending order"
        
        # model = DBSCAN(eps=0.01, min_samples=2)
        model = OPTICS(min_samples=2, xi=0.05)
        grouping = model.fit_predict(np.array(scores).reshape(-1, 1))
        return self._outlier_consolidate(grouping)


    def std_grouping(self, scores: List[float]):
        # assumed the scores are sorted decendingly
        sorted_scores = sorted(scores)[::-1]
        score_gaps = [sorted_scores[i]-sorted_scores[i+1] for i in range(len(sorted_scores)-1)]

        # calculate gap standard deviation for threshold
        threshold = np.std(score_gaps)

        # iterate for grouping
        current_score = sorted_scores[0]
        grouped_categories = [0]
        counter = 0
        for i, s in enumerate(sorted_scores):
            if i == 0:
                continue
            if abs(current_score - s) > threshold:
                counter += 1
            current_score = s
            grouped_categories.append(counter)
        return grouped_categories
    
    
    def consolidate_grouping(self, row):
        "To consolidate std and algo grouping"

        std_grouping = np.array(row['std_grouping'])
        algo_grouping = np.array(row['algo_grouping'])

        if np.sum(algo_grouping==0) <= 3:
            return algo_grouping
        if np.sum(algo_grouping==0) <= np.sum(std_grouping==0):
            return algo_grouping

        return std_grouping