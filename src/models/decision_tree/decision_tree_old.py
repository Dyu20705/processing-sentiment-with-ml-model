"""
Original Decision Tree implementation (before refactoring).

This is a backup version kept for reference and comparison.
Latest version: src/models/decision_tree/decision_tree.py
"""

from collections import Counter
import numpy as np


class Node:
    def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeOld:
    """Original Decision Tree implementation (simple version)."""
    
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _build_tree(self, dataset, curr_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape

        if num_samples >= self.min_samples_split and curr_depth < self.max_depth:
            best_split = self._get_best_split(dataset, num_features)
            if best_split["info_gain"] > 0:
                left_node = self._build_tree(best_split["left_dataset"], curr_depth + 1)
                right_node = self._build_tree(best_split["right_dataset"], curr_depth + 1)
                return Node(feature_idx=best_split["feature_idx"], threshold=best_split["threshold"],
                            info_gain=best_split["info_gain"], left=left_node, right=right_node)
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value=leaf_value)

    def _get_best_split(self, dataset, num_features):
        best_split = {"feature_idx": None, "threshold": None, "info_gain": -1, "left_dataset": None, "right_dataset": None}

        for feature_idx in range(num_features):
            feature_values = dataset[:, feature_idx]
            possible_thresholds = set(feature_values)

            for threshold in possible_thresholds:
                left_indices, right_indices = np.where(feature_values <= threshold)[0], np.where(feature_values > threshold)[0]
                if len(left_indices) > 0 and len(right_indices) > 0:
                    parent_y, left_y, right_y = dataset[:, -1], dataset[left_indices, -1], dataset[right_indices, -1]
                    info_gain = self._calculate_info_gain(parent_y, left_y, right_y)
                    if info_gain > best_split["info_gain"]:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["info_gain"] = info_gain
                        best_split["left_dataset"] = dataset[left_indices]
                        best_split["right_dataset"] = dataset[right_indices]
        return best_split
    
    def _calculate_info_gain(self, parent_y, left_y, right_y):
        weight_left = len(left_y) / len(parent_y)
        weight_right = len(right_y) / len(parent_y)

        gain = self._calculate_entropy(parent_y) - (weight_left * self._calculate_entropy(left_y) + weight_right * self._calculate_entropy(right_y))
        return gain
    
    def entropy(self, y):
        entropy = 0
        
        class_labels = np.unique(y)
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        return entropy
    
    def _calculate_entropy(self, y):
        """Calculate entropy of a label distribution."""
        if len(y) == 0:
            return 0
        
        entropy = 0.0
        class_labels = np.unique(y)
        
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            if p_cls > 0:
                entropy -= p_cls * np.log2(p_cls)
        
        return entropy

    def fit(self, X, y):
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self._build_tree(dataset)

    def predict(self, X):
        predictions = [self.predict_class(row, self.root) for row in X]
        return np.array(predictions)
    
    def predict_class(self, row, node):
        if node.value is not None:
            return node.value
        
        feature_value = row[node.feature_idx]
        if feature_value <= node.threshold:
            return self.predict_class(row, node.left)
        else:
            return self.predict_class(row, node.right)
