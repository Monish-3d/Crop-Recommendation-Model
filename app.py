import streamlit as st
import joblib
import numpy as np

model = joblib.load("crop_recommender_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

st.title("🌾 Crop Recommendation System")
st.write("Enter the environmental and soil parameters below to get a crop recommendation.")

nitrogen = st.number_input("Nitrogen (N) (Kg/ha)", min_value=0, max_value=200, value=50)
phosphorus = st.number_input("Phosphorus (P) (Kg/ha)", min_value=0, max_value=200, value=50)
potassium = st.number_input("Potassium (K) (Kg/ha)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, format="%.2f")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, format="%.2f")
ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=6.5, format="%.2f")
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, format="%.2f")

N_ratio = nitrogen / (nitrogen + phosphorus + potassium + 1e-9)
P_ratio = phosphorus / (nitrogen + phosphorus + potassium + 1e-9)
K_ratio = potassium / (nitrogen + phosphorus + potassium + 1e-9)
N_x_rain= nitrogen * rainfall

if st.button("Predict Crop"):
    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall,N_ratio,P_ratio,K_ratio,N_x_rain]])
    prediction_encoded = model.predict(features)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    
    st.success(f"Recommended Crop: **{prediction_label}**")

st.markdown("----------------------------")

