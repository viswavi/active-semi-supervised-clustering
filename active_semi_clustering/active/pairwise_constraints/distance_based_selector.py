import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate


class DistanceBasedSelector:
    def __init__(self, n_clusters=3, rerank_fraction = 0.005, **kwargs):
        self.n_clusters = n_clusters
        self.rerank_fraction = rerank_fraction

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


        remaining_dataset_indices = list(range(len(X)))

        # choose point pairs that are close together


        x_squared_norms = row_norms(X, squared=True)
        distance_matrix = euclidean_distances(X, X, Y_norm_squared=x_squared_norms, squared=True)
        distance_matrix_flattened = np.ravel(distance_matrix)
        flattened_matrix_sort_indices_unfiltered = np.argsort(distance_matrix_flattened)
        matrix_sort_indices_unfiltered = [(ind // len(X), ind % len(X)) for ind in flattened_matrix_sort_indices_unfiltered]
        matrix_sort_indices = [(x,y) for (x,y) in matrix_sort_indices_unfiltered if x < y and oracle.selected_sentences[x] != oracle.selected_sentences[y]]

        sampled_indices = []

        query_counter = 0
        while query_counter < oracle.max_queries_cnt / 2:
            query_counter += 1
            print(f"Query Counter: {query_counter}")
            new_pair = matrix_sort_indices[0]
            (x, y) = new_pair
            matrix_sort_indices = matrix_sort_indices[1:]
            assert len(matrix_sort_indices) > 0

            pair_label = oracle.query(x, y)
            if pair_label == True:
                ml.append([x, y])
            elif pair_label == False:
                cl.append([x, y])
            sampled_indices.extend([x, y])



        sampled_indices_arr = np.array(list(set(sampled_indices)))
        X_sampled = X[sampled_indices_arr]
        distance_matrix_from_seen = euclidean_distances(X, X_sampled, squared=True)
        max_distance_matrix_from_seen = np.max(distance_matrix_from_seen, axis=1)

        matrix_sort_indices_to_rerank = matrix_sort_indices[:int(self.rerank_fraction * len(matrix_sort_indices_unfiltered))]
        pair_distance_to_sampled = [min(max_distance_matrix_from_seen[i], max_distance_matrix_from_seen[j]) for i, j in matrix_sort_indices_to_rerank]
        pair_indices_reranked_by_descending_distance = np.argsort(-1 * np.array(pair_distance_to_sampled))
        matrix_indices_reranked_by_descending_distance = [matrix_sort_indices_to_rerank[tup] for tup in pair_indices_reranked_by_descending_distance]

        while query_counter < oracle.max_queries_cnt:
            query_counter += 1
            print(f"Query Counter: {query_counter}")
            new_pair = matrix_indices_reranked_by_descending_distance[0]
            (x, y) = new_pair
            matrix_indices_reranked_by_descending_distance = matrix_indices_reranked_by_descending_distance[1:]
            assert len(matrix_indices_reranked_by_descending_distance) > 0

            pair_label = oracle.query(x, y)

            if pair_label == True:
                ml.append([x, y])
            elif pair_label == False:
                cl.append([x, y])

        self.pairwise_constraints_ = (ml, cl)

        return self
