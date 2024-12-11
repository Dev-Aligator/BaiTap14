# 1. Import libraries
import pandas as pd
import numpy as np
import streamlit as st
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

# 6. Streamlit app
def predict_price(carat, depth, table, x, y, z, cut, color, clarity):
    # Create a single row of data for prediction
    input_data = pd.DataFrame({
        'carat': [carat],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z],
        **{f'cut_{cut}': [1] if f'cut_{cut}' in X.columns else [0] for cut in ['Good', 'Very Good', 'Premium', 'Ideal']},
        **{f'color_{color}': [1] if f'color_{color}' in X.columns else [0] for color in ['E', 'F', 'G', 'H', 'I', 'J']},
        **{f'clarity_{clarity}': [1] if f'clarity_{clarity}' in X.columns else [0] for clarity in ['SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL']}
    })
    # Fill missing columns with 0 if they do not exist in the model training data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Predict price
    return model.predict(input_data)[0]

st.title("Diamond Price Prediction")

# User inputs
carat = st.number_input("Carat", 0.0, 5.0, step=0.01)
depth = st.number_input("Depth", 0.0, 100.0, step=0.1)
table = st.number_input("Table", 0.0, 100.0, step=0.1)
x = st.number_input("X (length)", 0.0, 10.0, step=0.01)
y = st.number_input("Y (width)", 0.0, 10.0, step=0.01)
z = st.number_input("Z (depth)", 0.0, 10.0, step=0.01)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL'])

# Prediction button
if st.button("Predict Price"):
    price = predict_price(carat, depth, table, x, y, z, cut, color, clarity)
    st.success(f"Estimated price of the diamond: ${price:.2f}")
