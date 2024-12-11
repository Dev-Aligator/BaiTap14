import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown

model_url = "https://drive.google.com/uc?id=1GpMniUflvRXdosPnivsb3Y5daMhU62HH"

model_path = 'diamond_price_model.pkl'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Tạo giao diện người dùng với Streamlit
st.title("Dự đoán giá kim cương")

# Nhập các thông số của viên kim cương từ người dùng
carat = st.number_input("Carat", min_value=0.1, max_value=5.0, step=0.01)
cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Ideal', 'Excellent'])
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", ['I1', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2', 'IF'])

depth = st.number_input("Depth", min_value=50.0, max_value=80.0, step=0.1)
table = st.number_input("Table", min_value=50.0, max_value=70.0, step=0.1)
x = st.number_input("X (mm)", min_value=0.0, max_value=30.0, step=0.1)
y = st.number_input("Y (mm)", min_value=0.0, max_value=30.0, step=0.1)
z = st.number_input("Z (mm)", min_value=0.0, max_value=30.0, step=0.1)

# Chuyển đổi các thông số phân loại thành số (dùng LabelEncoder tương tự như ở phần huấn luyện mô hình)
label_encoder_cut = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Ideal': 3, 'Excellent': 4}
label_encoder_color = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
label_encoder_clarity = {'I1': 0, 'SI1': 1, 'SI2': 2, 'VS1': 3, 'VS2': 4, 'VVS1': 5, 'VVS2': 6, 'IF': 7}

cut_encoded = label_encoder_cut[cut]
color_encoded = label_encoder_color[color]
clarity_encoded = label_encoder_clarity[clarity]

# Dự đoán giá kim cương
if st.button("Dự đoán giá"):
    features = np.array([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]])
    predicted_price = model.predict(features)[0]
    st.success(f"Giá dự đoán của viên kim cương là: ${predicted_price:.2f} USD")
