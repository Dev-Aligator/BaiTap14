import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import pickle

# Tải mô hình và label_encoder từ Google Drive
model_url = 'https://drive.google.com/uc?id=1GxblzLJcfYtLH5kA7rwD0xKTPqku0HYU'
model_path = 'diamond_price_model.pkl'

# Tải tệp từ Google Drive
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("shubhankitsirvaiya06/diamond-price-prediction")

data_dir = os.path.join(path, 'diamonds.csv')
df = pd.read_csv(data_dir)

features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table']
df = df[features]

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['cut'] = label_encoder.fit_transform(df['cut'])
df['color'] = label_encoder.fit_transform(df['color'])
df['clarity'] = label_encoder.fit_transform(df['clarity'])

# Tạo giao diện Streamlit
st.title('Dự đoán giá kim cương')

# Nhận thông tin đầu vào từ người dùng
carat = st.number_input('Carat', min_value=0.01, max_value=5.0, value=1.0, step=0.01)
cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Clarity', ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])
depth = st.number_input('Depth', min_value=50.0, max_value=100.0, value=60.0, step=0.1)
table = st.number_input('Table', min_value=50.0, max_value=70.0, value=60.0, step=0.1)

# Mã hóa các giá trị đầu vào
cut_encoded = label_encoder.transform([cut])[0]
color_encoded = label_encoder.transform([color])[0]
clarity_encoded = label_encoder.transform([clarity])[0]

# Dự đoán giá kim cương
input_data = np.array([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table]])
predicted_price = model.predict(input_data)[0]

# Hiển thị kết quả
st.subheader(f'Giá kim cương dự đoán: ${predicted_price:,.2f}')
