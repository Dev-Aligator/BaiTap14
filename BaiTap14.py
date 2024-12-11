import pandas as pd
import numpy as np
import joblib
import streamlit as st
import gdown
import os
import pickle

model_url = "https://drive.google.com/uc?id=1SmqLZ3jPVYdeSAQAliTNX8SBymLf8MMZ"

model_path = 'diamond_price_model.pkl'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

def predict_price(carat, cut, color, clarity):
    cut_mapping = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Ideal': 3, 'Excellent': 4}
    color_mapping = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
    clarity_mapping = {'SI1': 0, 'VS1': 1, 'VVS1': 2, 'IF': 3, 'SI2': 4, 'VS2': 5, 'VVS2': 6}
    
    cut_num = cut_mapping[cut]
    color_num = color_mapping[color]
    clarity_num = clarity_mapping[clarity]
    
    input_data = np.array([carat, cut_num, color_num, clarity_num]).reshape(1, -1)
    
    predicted_price = model.predict(input_data)
    return predicted_price[0]

st.title("Ứng dụng dự đoán giá kim cương")
st.write("Nhập thông số về viên kim cương để dự đoán giá")

carat = st.slider('Trọng lượng kim cương (Carat)', 0.2, 5.0, 1.0)
cut = st.selectbox('Chất lượng cắt', ['Fair', 'Good', 'Very Good', 'Ideal', 'Excellent'])
color = st.selectbox('Màu sắc', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Độ trong suốt', ['SI1', 'VS1', 'VVS1', 'IF', 'SI2', 'VS2', 'VVS2'])

if st.button('Dự đoán giá'):
    predicted_price = predict_price(carat, cut, color, clarity)
    st.write(f"Giá kim cương dự đoán là: ${predicted_price:,.2f}")
