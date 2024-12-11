import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import gdown
import pickle
import os

# 2. Load dataset
model_url = "https://drive.google.com/uc?id=19z_sqUqgz9mUuGfqp9sfXTMTpHmcnyrg"
model_path = 'diamond_price_model.pkl'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
ordinal_encoder = OrdinalEncoder(categories=[cut_order])

st.title("Dự đoán Giá Kim Cương")

carat = st.slider("Carat", 0.1, 5.0, 1.0)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.slider("Depth (%)", 50.0, 70.0, 62.0)
table = st.slider("Table (%)", 50.0, 70.0, 57.0)

cut_encoded = ordinal_encoder.transform([[cut]])[0][0]

input_data = np.array([[carat, depth, table, cut_encoded]])

scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

predicted_price = model.predict(input_data_scaled)[0]

st.subheader(f"Giá kim cương dự đoán: ${predicted_price:,.2f}")
