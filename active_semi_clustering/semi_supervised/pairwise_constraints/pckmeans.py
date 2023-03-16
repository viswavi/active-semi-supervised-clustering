import json
import numpy as np

from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints
from active_semi_clustering.semi_supervised.labeled_data.kmeans import KMeans

class PCKMeans(KMeans):
    def __init__(self, n_clusters=3, max_iter=100, w=1, init="random"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w
        self.init = init

    def fit(self, X, y=None, ml=[], cl=[]):
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        print(f"ML constraints:\n{ml}\n")
        print(f"CL constraints:\n{cl}\n")

        print(f"Num neighborhoods: {neighborhoods}\n\n\n")

        # Initialize centroids
        # cluster_centers = self._init_cluster_centers(X)
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            print(f"\n\n\n\niteration: {iteration}")
            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters:
                if self.init == "k-means++":
                    if len(list(cluster_centers)) > 0:
                        cluster_centers = super()._init_cluster_centers(X, seed_set=list(cluster_centers))
                    else:
                        cluster_centers = super()._init_cluster_centers(X)
                else:
                    remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                    cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])
        return cluster_centers

    def _objective_function(self, X, x_i, centroids, c_i, labels, ml_graph, cl_graph, w, print_terms=False):
        distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)

        ml_penalty = 0
        for y_i in ml_graph[x_i]:
            if labels[y_i] != -1 and labels[y_i] != c_i:
                ml_penalty += w

        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                cl_penalty += w
        if print_terms:
            metric_dict = {"x_i": x_i, "distance": round(distance, 4), "ml_penalty": round(ml_penalty, 4), "cl_penalty": round(ml_penalty, 4)}
            print(json.dumps(metric_dict))

        return distance + ml_penalty + cl_penalty

    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)
        min_cluster_distances = []

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for x_i in index:
            cluster_distances = [self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w) for c_i in range(self.n_clusters)]
            min_cluster_distances.append(min(cluster_distances))
            labels[x_i] = np.argmin(cluster_distances)

            _ = self._objective_function(X, x_i, cluster_centers, labels[x_i], labels, ml_graph, cl_graph, w, print_terms=True)


        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            original_labels = labels.copy()
            print(f"Empty clusters: {empty_clusters}")
            points_by_min_cluster_distance = np.argsort(-np.array(min_cluster_distances))
            i = 0
            for cluster_idx in list(empty_clusters):
                while n_samples_in_cluster[labels[points_by_min_cluster_distance[i]]] == 1:
                    i += 1
                labels[points_by_min_cluster_distance[i]] = cluster_idx
                i += 1

            n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
            empty_clusters = np.where(n_samples_in_cluster == 0)[0]
            if len(empty_clusters) > 0:
                breakpoint()
        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
