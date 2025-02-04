import numpy as np

class SVMFromScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.classifiers = []
        self.classes = None 

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        for class_label in self.classes:
            y_binary = np.where(y == class_label, 1, -1)

            n_samples, n_features = X.shape
            weights = np.zeros(n_features)
            bias = 0

            for _ in range(self.epochs):
                for idx, x_i in enumerate(X):
                    condition = y_binary[idx] * (np.dot(x_i, weights) - bias) >= 1
                    if condition:
                        weights -= self.learning_rate * (2 * self.lambda_param * weights)
                    else:
                        weights -= self.learning_rate * (2 * self.lambda_param * weights - np.dot(x_i, y_binary[idx]))
                        bias -= self.learning_rate * y_binary[idx]

            self.classifiers.append((weights, bias))

    def predict(self, X):
        decision_scores = np.zeros((X.shape[0], len(self.classes)))

        for i, (weights, bias) in enumerate(self.classifiers):
            decision_scores[:, i] = np.dot(X, weights) - bias

        return self.classes[np.argmax(decision_scores, axis=1)]

    def decision_function(self, X):
        decision_scores = np.zeros((X.shape[0], len(self.classes)))

        for i, (weights, bias) in enumerate(self.classifiers):
            decision_scores[:, i] = np.dot(X, weights) - bias

        return decision_scores