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

trained_features = [
    'carat', 'cut_Fair', 'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good',
    'color_D', 'color_E', 'color_F', 'color_G', 'color_H', 'color_I', 'color_J',
    'clarity_I1', 'clarity_IF', 'clarity_SI1', 'clarity_SI2', 
    'clarity_VS1', 'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2',
    'depth', 'table', 'x', 'y', 'z'
]

def preprocess_input(carat, cut, color, clarity, depth, table, x, y, z):
    # Mã hóa các giá trị phân loại
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut_Fair': [1 if cut == 'Fair' else 0],
        'cut_Good': [1 if cut == 'Good' else 0],
        'cut_Ideal': [1 if cut == 'Ideal' else 0],
        'cut_Premium': [1 if cut == 'Premium' else 0],
        'cut_Very Good': [1 if cut == 'Very Good' else 0],
        'color_D': [1 if color == 'D' else 0],
        'color_E': [1 if color == 'E' else 0],
        'color_F': [1 if color == 'F' else 0],
        'color_G': [1 if color == 'G' else 0],
        'color_H': [1 if color == 'H' else 0],
        'color_I': [1 if color == 'I' else 0],
        'color_J': [1 if color == 'J' else 0],
        'clarity_I1': [1 if clarity == 'I1' else 0],
        'clarity_IF': [1 if clarity == 'IF' else 0],
        'clarity_SI1': [1 if clarity == 'SI1' else 0],
        'clarity_SI2': [1 if clarity == 'SI2' else 0],
        'clarity_VS1': [1 if clarity == 'VS1' else 0],
        'clarity_VS2': [1 if clarity == 'VS2' else 0],
        'clarity_VVS1': [1 if clarity == 'VVS1' else 0],
        'clarity_VVS2': [1 if clarity == 'VVS2' else 0],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })

    input_data = input_data.reindex(columns=trained_features, fill_value=0)
    return input_data

def predict_price(carat, cut, color, clarity, depth, table, x, y, z):
    input_data = preprocess_input(carat, cut, color, clarity, depth, table, x, y, z)
    
    predicted_price = model.predict(input_data)
    return predicted_price[0]
    
st.title("Ứng dụng dự đoán giá kim cương")
st.write("Nhập thông số về viên kim cương để dự đoán giá")

carat = st.slider('Trọng lượng kim cương (Carat)', 0.2, 5.0, 1.0)
cut = st.selectbox('Chất lượng cắt', ['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'])
color = st.selectbox('Màu sắc', ['D', 'E', 'F', 'G', 'H'])
clarity = st.selectbox('Độ trong suốt', ['I1', 'SI1', 'VS1', 'VVS1'])
depth = st.number_input('Độ sâu (%)', 50.0, 70.0, 60.0)
table = st.number_input('Bảng (%)', 50.0, 75.0, 60.0)
x = st.number_input('Chiều dài (mm)', 0.0, 10.0, 5.0)
y = st.number_input('Chiều rộng (mm)', 0.0, 10.0, 5.0)
z = st.number_input('Chiều cao (mm)', 0.0, 10.0, 3.0)

if st.button('Dự đoán giá'):
    predicted_price = predict_price(carat, cut, color, clarity, depth, table, x, y, z)
    st.write(f"Giá kim cương dự đoán là: ${predicted_price:,.2f}")
