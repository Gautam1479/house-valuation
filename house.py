import streamlit as st
import joblib  # pip install joblib
from sklearn.preprocessing import StandardScaler as ss
import pandas as pd
import pickle

# Streamlit app configuration
st.set_page_config("House Valuation")
st.title("House Price Calculator")

# Input fields (they return strings, so we need to convert them to numeric types later)
val1 = st.text_input("Area", placeholder="Enter the area size in square feet")
val2 = st.text_input("Bedrooms", placeholder="Enter the number of bedrooms")
val3 = st.text_input("Bathrooms", placeholder="Enter the number of bathrooms")
val4 = st.text_input("Stories", placeholder="Enter the number of stories")
val5 = st.text_input("Parking", placeholder="Enter the number of extra parking")

# Radio buttons for Yes/No features
has_mainroad = st.radio('Has Mainroad?', ['Yes', 'No'])
has_guestroom = st.radio('Has Guestroom?', ['Yes', 'No'])
has_basement = st.radio('Has Basement?', ['Yes', 'No'])
has_aircondi = st.radio('Has Airconditioning?', ['Yes', 'No'])
has_prefarea = st.radio('Location in Preferred Area?', ['Yes', 'No'])
has_furniture = st.radio('Furnished or not?', ['Furnished', 'Semi-Furnished', 'Unfurnished'])
has_mainroad = 1 if has_mainroad == 'Yes' else 0
has_guestroom = 1 if has_guestroom == 'Yes' else 0
has_basement = 1 if has_basement == 'Yes' else 0
has_aircondi = 1 if has_aircondi == 'Yes' else 0
has_prefarea = 1 if has_prefarea == 'Yes' else 0
if has_furniture == 'Furnished':
    furniture_numeric = 2
elif has_furniture == 'Semi-Furnished':
    furniture_numeric = 1
else:
    furniture_numeric = 0
df = pd.read_csv("Housing.csv")
df.drop_duplicates(inplace=True)
df.drop(labels=["hotwaterheating"], axis=1, inplace=True)
scaler = ss()
scaler.fit(df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']])
with open('Housing.pkl', 'rb') as file:
    model = pickle.load(file)

if st.button('Predict'):
    try:
        val1 = float(val1)
        val2 = int(val2)
        val3 = int(val3)
        val4 = int(val4)
        val5 = int(val5)
    except ValueError:
        st.error("Please enter valid numeric values!")
        st.stop()
import numpy as np
scaled_vals = scaler.transform([[val1, val2, val3, val4, val5]])
features =np.array([scaled_vals,has_mainroad, has_guestroom, has_basement, has_aircondi, has_prefarea, furniture_numeric])
prediction = model.predict([features])
st.write(f"Predicted House Price: ${prediction[0]:,.2f}")   