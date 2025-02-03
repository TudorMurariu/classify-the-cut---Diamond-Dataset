import pickle
import numpy as np
import pandas as pd

with open('stacking_model.pkl', 'rb') as f:
    models = pickle.load(f)
    meta_model = pickle.load(f)

input_data = {
    'cut': ['Ideal'],
    'price': [2757],
    'carat': [0.75],
    'x': [5.83],
    'y': [5.87],
    'z': [3.64],
    'depth': [62.2],
    'table': [55.0],
    'color': ['D'],
    'clarity': ['SI2']
}

input_df = pd.DataFrame(input_data)

color_mapping = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
clarity_mapping = {'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7}
cut_mapping = {'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4}

input_df['color'] = input_df['color'].map(color_mapping)
input_df['clarity'] = input_df['clarity'].map(clarity_mapping)
input_df['cut'] = input_df['cut'].map(cut_mapping)

features = ['price', 'carat', 'x', 'y', 'z', 'depth', 'color', 'table', 'clarity']
input_array = input_df[features].values

meta_features = np.zeros((input_array.shape[0], len(models)))

for i, model in enumerate(models):
    meta_features[:, i] = model.predict(input_array)

final_prediction = meta_model.predict(meta_features)

reverse_cut_mapping = {v: k for k, v in cut_mapping.items()}
predicted_cut = reverse_cut_mapping[final_prediction[0]]

print("Predicted Cut Category:", predicted_cut)