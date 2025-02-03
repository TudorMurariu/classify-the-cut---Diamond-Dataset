import numpy as np

class GaussianNBFromScratch:
    def __init__(self):
        self.class_probs = None
        self.class_means = None
        self.class_vars = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.class_probs = {}
        self.class_means = {}
        self.class_vars = {}

        # Calculate class-wise statistics
        for class_label in np.unique(y):
            X_class = X[y == class_label]
            self.class_probs[class_label] = X_class.shape[0] / n_samples
            self.class_means[class_label] = np.mean(X_class, axis=0)
            self.class_vars[class_label] = np.var(X_class, axis=0)

    def _gaussian_likelihood(self, x, class_label):
        mean = self.class_means[class_label]
        var = self.class_vars[class_label]

        # Add a small epsilon to the variance to avoid division by zero or log(0)
        epsilon = 1e-10
        variance_adjusted = np.maximum(var, epsilon)

        # Compute the likelihood of the feature values (Gaussian distribution)
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / variance_adjusted)
        likelihood = (1 / np.sqrt(2 * np.pi * variance_adjusted)) * exponent
        
        # Prevent log(0) by adding a small value to likelihood if it's zero
        likelihood = np.maximum(likelihood, epsilon)

        return likelihood

    def predict(self, X):
        # Calculate log-probabilities for each class
        log_probs = []

        for x in X:
            class_log_probs = {}
            for class_label in self.class_probs:
                log_prob = np.log(self.class_probs[class_label])  # Prior
                likelihood = self._gaussian_likelihood(x, class_label)
                log_prob += np.sum(np.log(likelihood))  # Likelihoods

                class_log_probs[class_label] = log_prob

            # Predict class with highest log-probability
            predicted_class = max(class_log_probs, key=class_log_probs.get)
            log_probs.append(predicted_class)

        return np.array(log_probs)

