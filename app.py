import streamlit as st
import numpy as np
import pickle

# Load the model from the pickle file
def load_model():
    with open('xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
xgb_model = load_model()

# Define the number of input features expected by your model
input_feature_length = 30  # Update this value to match the number of features expected by your model

# Streamlit app

st.markdown("<h2 style='text-align: center;'>Institute of Technology of Cambodia</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Department of AMS</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Credit Card Fraud Detection 🕵️</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Made By Group-5</h3>", unsafe_allow_html=True)
st.write('Please input all required features with comma-separated values(Time, V1, V2, ..., V28, Amount)')
# Input from user
input_features = st.text_input('Enter the features (comma-separated)')

# Submit button
submit = st.button('Submit')

if submit:
    try:
        features = np.array(input_features.split(','), dtype=np.float64).reshape(1, -1)
        if features.shape[1] != input_feature_length:
            st.error(f"Expected {input_feature_length} features, but got {features.shape[1]}")
        else:
            prediction = xgb_model.predict(features)

            if prediction[0] == 0:
                st.success('The transaction is legitimate!')
            else:
                st.error('The transaction is a fraud!')
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")
