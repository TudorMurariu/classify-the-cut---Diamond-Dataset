import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from LogisticRegressor import LogisticRegressionScratch
from SVM import SVMFromScratch

df = pd.read_csv("DiamondsPrices.csv")

color_mapping = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
clarity_mapping = {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7}

df['color'] = df['color'].map(color_mapping)
df['clarity'] = df['clarity'].map(clarity_mapping)

cut_mapping = {'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4}
df['cut'] = df['cut'].map(cut_mapping)

X = df[['price', 'carat', 'x', 'y', 'z', 'depth', 'color', 'table', 'clarity']].values
y = df['cut'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    LogisticRegressionScratch(learning_rate=0.01, epochs=150),
    SVMFromScratch(learning_rate=0.001, lambda_param=0.01, epochs=400),
    SVMFromScratch(learning_rate=0.0008, lambda_param=0.02, epochs=700),
    SVMFromScratch(learning_rate=0.0015, lambda_param=0.005, epochs=200)
]

train_meta_features = np.zeros((X_train.shape[0], len(models)))
test_meta_features = np.zeros((X_test.shape[0], len(models)))

for i, model in enumerate(models):
    model.fit(X_train, y_train)

    train_meta_features[:, i] = model.predict(X_train)
    test_meta_features[:, i] = model.predict(X_test)

    model_predictions = model.predict(X_test)
    model_accuracy = accuracy_score((y_test == i).astype(int), model_predictions)
    print(f"Model {i + 1} Accuracy: {model_accuracy:.4f}")

meta_model = SVMFromScratch(learning_rate=0.001, lambda_param=0.01, epochs=100)
meta_model.fit(train_meta_features, y_train)

final_predictions = meta_model.predict(test_meta_features)

accuracy = accuracy_score(y_test, final_predictions)
print("Stacking Model Accuracy:", accuracy)

import pickle

# Save the entire stacking model (including base models) to a file
with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(models, f)  # Save the base models
    pickle.dump(meta_model, f)  # Save the meta model


# import joblib
# joblib.dump((models, meta_model), 'stacking_model.joblib')