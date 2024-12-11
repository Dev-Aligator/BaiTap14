import pandas as pd
import numpy as np
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

def preprocess_input(carat, cut, color, clarity):
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut_Fair': [1 if cut == 'Fair' else 0],
        'cut_Good': [1 if cut == 'Good' else 0],
        'cut_Ideal': [1 if cut == 'Ideal' else 0],
        'cut_Excellent': [1 if cut == 'Excellent' else 0],
        'cut_Very Good': [1 if cut == 'Very Good' else 0],
        'color_D': [1 if color == 'D' else 0],
        'color_E': [1 if color == 'E' else 0],
        'color_F': [1 if color == 'F' else 0],
        'color_G': [1 if color == 'G' else 0],
        'color_H': [1 if color == 'H' else 0],
        'color_I': [1 if color == 'I' else 0],
        'color_J': [1 if color == 'J' else 0],
        'clarity_IF': [1 if clarity == 'IF' else 0],
        'clarity_SI1': [1 if clarity == 'SI1' else 0],
        'clarity_SI2': [1 if clarity == 'SI2' else 0],
        'clarity_VS1': [1 if clarity == 'VS1' else 0],
        'clarity_VS2': [1 if clarity == 'VS2' else 0],
        'clarity_VVS1': [1 if clarity == 'VVS1' else 0],
        'clarity_VVS2': [1 if clarity == 'VVS2' else 0]
    })
    return input_data

def predict_price(carat, cut, color, clarity):
    input_data = preprocess_input(carat, cut, color, clarity)
    
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
