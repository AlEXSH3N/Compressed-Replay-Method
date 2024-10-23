from tqdm import tqdm
from skopt import Optimizer
from skopt.space import Integer
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import numpy as np
import torch


class KMeansOptimizer:
    def __init__(self, features, targets, loss_values, input_shape, n_calls=20, max_retrieval=2000, random_state=42):
        # Flatten features based on input_shape
        self.input_shape = input_shape
        if isinstance(features, torch.Tensor):
            features = features.view(features.size(0), -1).cpu().numpy()
        elif features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        self.features = features
        self.loss_values = np.array(loss_values)
        self.targets = np.array([t.item() for t in targets])
        self.n_calls = n_calls
        self.max_retrieval = max_retrieval
        self.random_state = random_state
        self.optimizer = Optimizer([
            Integer(2, min(15, len(self.features) // 10), name='num_clusters'),
            Integer(300, 1000, name='instances_per_cluster')
        ], random_state=random_state)

    def run_optimization(self):
        best_score = -float('inf')
        best_params = None
        best_cluster_labels = None

        for _ in tqdm(range(self.n_calls), desc="Optimizing K-means"):
            params = self.optimizer.ask()
            score, cluster_labels = self._objective(params)
            if cluster_labels is None:
                continue  # Skip this iteration if clustering failed

            if not np.isinf(score):
                self.optimizer.tell(params, -score)

            if score > best_score:
                best_score = score
                best_params = params
                best_cluster_labels = cluster_labels
                print(f"New best score: {best_score}")

        if best_params is None:
            raise ValueError("No valid clustering found during optimization.")
        
        selected_indices = self._select_stratified_samples(best_cluster_labels, best_params[1])
        return selected_indices, best_params


    def _objective(self, params):
        num_clusters, instance_per_clusters = params

        # Dynamically reduce clusters if data is very similar
        num_clusters = min(num_clusters, len(np.unique(self.features, axis=0)))

        kmeans = KMeans(n_clusters=num_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(self.features)

        # Handle cases where KMeans might not find enough clusters
        if len(set(cluster_labels)) < 2:
            return -float('inf'), None

        try:
            score = silhouette_score(self.features, cluster_labels)
        except ValueError as e:
            print(f"Error calculating silhouette score: {e}")
            return -float('inf'), None

        return score, cluster_labels


    def _select_stratified_samples(self, cluster_labels, instances_per_cluster):
        unique_labels = np.unique(self.targets)
        selected_indices = set()
        number_list = []

        for label in unique_labels:
            indices = np.where(self.targets == label)[0]
            label_cluster_labels = cluster_labels[indices]
            label_loss_values = self.loss_values[indices]

            for cluster in np.unique(label_cluster_labels):
                cluster_indices = np.where(label_cluster_labels == cluster)[0]
                cluster_losses = label_loss_values[cluster_indices]
                sorted_loss_indices = cluster_indices[np.argsort(cluster_losses)]

                half_split = instances_per_cluster
                lowest_loss_indices = sorted_loss_indices[:half_split]
                # median_start = len(sorted_loss_indices) // 4
                # median_loss_indices = sorted_loss_indices[median_start: median_start + half_split]
                
                # Add indices to the set to ensure uniqueness
                selected_indices.update(indices[lowest_loss_indices].tolist())
                # selected_indices.update(indices[median_loss_indices].tolist())
                
            number_list.append(len(lowest_loss_indices))

        selected_indices = np.array(list(selected_indices))  # Convert back to numpy array
        selected_indices = shuffle(selected_indices, random_state=self.random_state)
        selected_indices = selected_indices[:min(len(selected_indices), self.max_retrieval)]

        print(f"Selected {len(selected_indices)} samples after balancing.")
        print(f"All elements in selected_indices are unique: {len(selected_indices) == len(set(selected_indices))}")
        print(f"Number of samples selected per class: {number_list}")
        
        return selected_indices
