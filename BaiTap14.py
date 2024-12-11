import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown
import pickle
import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shubhankitsirvaiya06/diamond-price-prediction")

print("Path to dataset files:", path)

data_dir = os.path.join(path, 'diamonds.csv')
data = pd.read_csv(data_dir)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
encoder = OrdinalEncoder(categories=[cut_order])
data['cut'] = encoder.fit_transform(data[['cut']])

data = pd.get_dummies(data, columns=['color', 'clarity'], drop_first=True)

from sklearn.preprocessing import StandardScaler

numeric_features = ['carat', 'depth', 'table']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])



model_url = "https://drive.google.com/uc?id=1iwG4azuPyAFPbcmcXyUC95CS4CWcD_qL"
model_path = 'diamond_price_model.pkl'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title("Ứng dụng dự đoán giá kim cương")

carat = st.slider("Carat", 0.1, 5.0, 1.0)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.slider("Depth (%)", 50.0, 70.0, 62.0)
table = st.slider("Table (%)", 50.0, 70.0, 57.0)

cut_encoded = encoder.transform([[cut]])[0][0]

input_data = np.array([[carat, depth, table, cut_encoded]])

input_data_scaled = scaler.transform(input_data)

predicted_price = model.predict(input_data_scaled)[0]

st.subheader(f"Giá kim cương dự đoán: ${predicted_price:,.2f}")
