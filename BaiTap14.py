import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown
import pickle
import os

model_url = "https://drive.google.com/uc?id=1iwG4azuPyAFPbcmcXyUC95CS4CWcD_qL"
model_path = 'diamond_price_model.pkl'

encoder_url = "https://drive.google.com/uc?id=1TiMCkR623_zSodT1QXErC6HLrzN8kYjA"
encoder_path = 'ordinal_encoder.pkl'

scaler_url = "https://drive.google.com/uc?id=1i6Vasx1zMK0LAKgIQRQBj1tN0-86Ayu3"
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)
with open(model_path, 'rb') as file:
    model = pickle.load(file)


if not os.path.exists(encoder_path):
    gdown.download(encoder_url, encoder_path, quiet=False)
with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)


if not os.path.exists(scaler_path):
    gdown.download(scaler_url, scaler_path, quiet=False)
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

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
