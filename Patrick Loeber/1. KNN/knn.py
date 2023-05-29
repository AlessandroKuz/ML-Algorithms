"""The KNN algorithm is a simple, yet effective, algorithm that can be used to solve
classification and regression problems. The KNN algorithm is a type of supervised
machine learning, which means that we are given a labelled dataset. In other words,
each data point is annotated with its corresponding label. The KNN algorithm assumes
that similar things exist in close proximity. In other words, similar things are near to
each other."""

import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k:int=3) -> None:
        self.k = k

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        self.X_train = X
        self.y_train = y

    def predict(self, X:np.ndarray):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x:np.ndarray):
        # compute distances
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


    @staticmethod
    def euclidean_distance(x1:np.ndarray, x2:np.ndarray):
        return np.sqrt(np.sum((x1 - x2)**2))
