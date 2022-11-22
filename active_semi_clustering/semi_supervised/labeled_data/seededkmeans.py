from collections import defaultdict
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
import sys

from .kmeans import KMeans

from cmvc.Multi_view_CH_kmeans import init_seeded_kmeans_plusplus


class SeededKMeans(KMeans):
    def _init_cluster_centers(self, X, y=None, random_seed=0):
        assert y is not None and not np.all(y == -1)
        random_state = check_random_state(random_seed)
        x_squared_norms = row_norms(X, squared=True)

        seed_set = [i for i, y_value in enumerate(y) if y_value != -1]
        if self.init == "k-means++":
            seeds = init_seeded_kmeans_plusplus(X, seed_set, self.n_clusters, x_squared_norms, random_state)
        else:
            remaining_seeds_available = list(set(range(len(X))) - set(seed_set))
            remaining_seeds_chosen = np.random.choice(remaining_seeds_available, size=self.n_clusters - len(seed_set), replace=False)
            seeds = X[np.concatenate([seed_set, remaining_seeds_chosen])]
        return seeds
