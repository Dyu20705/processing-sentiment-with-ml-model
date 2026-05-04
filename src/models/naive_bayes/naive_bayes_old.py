"""
Original Naive Bayes implementation (before refactoring).

This is a backup version kept for reference and comparison.
Latest version: src/models/naive_bayes/naive_bayes.py
"""

import numpy as np


class NaiveBayesOld:
    """Original Naive Bayes implementation (simple version)."""
    
    def __init__(self):
        # state của model
        pass

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.priors = [len(y_train[y_train == c]) / len(y_train) for c in self.classes]

        self.means = [X_train[y_train == c].mean(axis=0) for c in self.classes]
        self.stds = [X_train[y_train == c].std(axis=0) for c in self.classes]

    def compute_likelihood(self, row, class_idx):
        likelihood = 1
        for feature in row.index:
            mean = self.means[class_idx][feature]
            std = self.stds[class_idx][feature]
            likelihood *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((row[feature] - mean) / std) ** 2)
        return likelihood
    
    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            posteriors = []
            for i in range(len(self.classes)):
                likelihood = self.compute_likelihood(row, i)
                posteriors.append(likelihood * self.priors[i])

            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)
