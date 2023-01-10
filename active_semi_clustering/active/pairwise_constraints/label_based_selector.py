import numpy as np

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate


class LabelBasedSelector:
    def __init__(self, n_clusters=3, **kwargs):
        self.n_clusters = n_clusters

    def choose_closest_point(self, X, sample_idx, other_indices):
        distances = np.linalg.norm(X[sample_idx] - X[other_indices], axis=1)
        return other_indices[np.argmin(distances)]

    def choose_furthest_point(self, X, sample_idx, other_indices):
        distances = np.linalg.norm(X[sample_idx] - X[other_indices], axis=1)
        return other_indices[np.argmax(distances)]

    def fit(self, X, oracle=None):
        if oracle.max_queries_cnt <= 0:
            return [], []

        labels = oracle.labels

        ml = []
        cl = []


        dataset_indices = []
        clusters_to_idxs = {}
        idxs_to_clusters = {}
        for idx, l in enumerate(labels):
            dataset_indices.append(idx)
            if l not in clusters_to_idxs:
                clusters_to_idxs[l] = []
            clusters_to_idxs[l].append(idx)
            idxs_to_clusters[idx] = l
        remaining_dataset_indices = dataset_indices


        while len(ml) + len(cl) < oracle.max_queries_cnt:
            sample_idx = np.random.choice(dataset_indices)
            cluster_label = idxs_to_clusters[sample_idx]
            clusters_to_idxs[cluster_label].remove(sample_idx)
            if len(clusters_to_idxs[cluster_label]) == 0:
                continue

            remaining_dataset_indices.remove(sample_idx)

            in_cluster_idxs = clusters_to_idxs[cluster_label]
            out_of_cluster_idxs = []
            for l in clusters_to_idxs:
                if l is not cluster_label:
                    out_of_cluster_idxs.extend(clusters_to_idxs[l])

            furthest_in_cluster_point_idx = self.choose_furthest_point(X, sample_idx, in_cluster_idxs)
            closest_out_of_cluster_point_idx = self.choose_closest_point(X, sample_idx, out_of_cluster_idxs)

            cl.append([sample_idx, closest_out_of_cluster_point_idx])
            ml.append([sample_idx, furthest_in_cluster_point_idx])

        self.pairwise_constraints_ = (ml, cl)

        return self