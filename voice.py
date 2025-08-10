import joblib
import streamlit as st
import numpy as np

# Import the model
scaler = joblib.load("scaler.pkl")
model = joblib.load("svc_model.pkl")

# Title for UI
st.title("🎙️ Human Voice Classfication")

Inputs = ['mfcc_5_mean(-50,20)',
          'mean_spectral_contrast(15,25)',
          'mfcc_3_std(15,65)',
          'mfcc_2_mean(10,150)',
          'std_spectral_bandwidth(200,700)',
          'mfcc_12_mean(-15,15)',
          'mfcc_1_mean(-450,-150)',
          'mfcc_10_mean(-20,20)',
          'rms_energy(0,0.2)',
          'mfcc_10_std(5,20)',
          'mfcc_2_std(20,100)',
          'mfcc_8_mean(-30,20)',
          'mfcc_6_mean(-25,25)',
          'mfcc_4_mean(-10,100)',
          'mfcc_13_mean(-25,5)']

st.subheader("Enter the values of features")
cols = st.columns(3)

data = [None] * len(Inputs)

for i,input in enumerate(Inputs):
    index = i % 3

    with cols[index]:
        val = st.number_input(input)
        data[i] = val


if st.button("🔍 Predict Gender", use_container_width=True):

    data_scale = scaler.transform([data])
    gender = model.predict(data_scale)

    if gender.item() == 1:
        st.success("🧑 It's a **Male** voice.")
    else:
        st.success("👩 It's a **Female** voice.")