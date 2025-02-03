import numpy as np

class KNNClassifierScratch:
    def __init__(self, k=3, sample_size=None):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.sample_size = sample_size  # Percentage of training data to sample 

    def fit(self, X, y):
        if self.sample_size:
            sample_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * self.sample_size), replace=False)
            self.X_train = X[sample_indices]
            self.y_train = y[sample_indices]
        else:
            self.X_train = X
            self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = np.array([self.euclidean_distance(x, x_train) for x_train in self.X_train])

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = self.y_train[k_indices]

        return np.bincount(k_nearest_labels).argmax()
