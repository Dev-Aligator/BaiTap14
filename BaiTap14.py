import streamlit as st
import pandas as pd
import numpy as np
import gdown
import joblib

# Tải mô hình và label_encoder từ Google Drive
model_url = 'https://drive.google.com/uc?id=1UtVhI3XtBR-vhYMrnCkCsdcSRXot2dS4'
label_encoder_url = 'https://drive.google.com/uc?id=1BAaGBA7VgzgUthPVrlrPrVYuL3BaVtmD'

# Tải tệp từ Google Drive
gdown.download(model_url, 'model.pkl', quiet=False)
gdown.download(label_encoder_url, 'label_encoder.pkl', quiet=False)

# Tải mô hình và label_encoder từ tệp .pkl
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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
