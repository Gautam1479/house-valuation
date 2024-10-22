import numpy as np
import streamlit as st
# import joblib  # pip install joblib
from sklearn.preprocessing import StandardScaler as ss
import pandas as pd
import pickle
import joblib
# 
# Streamlit app configuration
st.set_page_config("House Valuation")
st.title("House Price Calculator")

# Input fields (they return strings, so we need to convert them to numeric types later)
val1 = st.number_input("Area", placeholder="Enter the area size in square feet")
val2 = st.number_input("Bedrooms", placeholder="Enter the number of bedrooms")
val3 = st.number_input("Bathrooms", placeholder="Enter the number of bathrooms")
val4 = st.number_input("Stories", placeholder="Enter the number of stories")
val5 = st.number_input("Parking", placeholder="Enter the number of extra parking")

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
    unfurnished=0
    semi_furniture = 0
elif has_furniture == 'Semi-Furnished':
    semi_furniture = 1
    unfurnished=0
else:
    semi_furniture = 0
    unfurnished=1
df = pd.read_csv("Housing.csv")
df.drop_duplicates(inplace=True)
df.drop(labels=["hotwaterheating"], axis=1, inplace=True)
scaler = ss()
scaler.fit(df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']])
# with open('Housing.pkl', 'rb') as file:
#     model = pickle.load(file)
model=joblib.load('Housing.pkl')
if st.button('Predict'):
    scaled_vals = scaler.transform([[val1, val2, val3, val4, val5]])
    scaled_vals = scaled_vals.flatten()
# Create the feature array by concatenating flattened scaled_vals and other features
    features = np.array([*scaled_vals, has_mainroad, has_guestroom, has_basement, has_aircondi, has_prefarea,semi_furniture, unfurnished])
    # st.write(features)
    prediction = model.predict([features])
    st.write(f"Predicted House Price: ${prediction[0]:,.2f}")   
