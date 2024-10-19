import streamlit as st
import joblib # pip install joblib
from sklearn.preprocessing import StandardScaler as ss
st.set_page_config("House valuation")
st.title("House price calculator")
val1=st.text_input("area",placeholder="Enter the area size in square feet")
val2=st.text_input("bedrooms",placeholder="Enter the number of bedrooms")
val3=st.text_input("bathrooms",placeholder="Enter the number of bathrooms")
val4=st.text_input("stories",placeholder="Enter the number of stories")
val5=st.text_input("parking",placeholder="Enter the number of extra parking")
has_mainroad = st.radio('Has Mainroad?', ['Yes', 'No'])
has_guestroom = st.radio('Has Guestroom?', ['Yes', 'No'])
has_basement = st.radio('Has Basement?', ['Yes', 'No'])
has_aircondi = st.radio('Has Airconditioning?', ['Yes', 'No'])
has_prefarea = st.radio('Location in Prefered Area?', ['Yes', 'No'])
has_furniture = st.radio('Furnished or not?', ['Yes', 'No'])
# 'price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
#        'mainroad_yes', 'guestroom_yes', 'basement_yes', 'airconditioning_yes',
#        'prefarea_yes', 'furnishingstatus_semi-furnished',
#        'furnishingstatus_unfurnished'
has_mainroad = 1 if has_mainroad == 'Yes' else 0
has_guestroom = 1 if has_guestroom == 'Yes' else 0
has_basement = 1 if has_basement== 'Yes' else 0
has_aircondi = 1 if has_aircondi == 'Yes' else 0
has_prefarea = 1 if has_prefarea == 'Yes' else 0
has_furniture = 1 if has_furniture == 'Yes' else 0
model = joblib.load('Housing.pkl')
features=[val1,val2,val3,val4,val5,has_mainroad,has_guestroom,has_basement,has_aircondi,has_prefarea,has_furniture]

import pandas as pd

df=pd.read_csv("Housing.csv")
df.drop_duplicates
df.drop(labels=["hotwaterheating"],axis=1,inplace=True)
scaler=ss()
scaler.fit(df.loc[:,['area', 'bedrooms', 'bathrooms', 'stories', 'parking']])
import pickle

# Load the model from the .pkl file
with open('Housing.pkl', 'rb') as file:
    model = pickle.load(file)
st.write(model)

if st.button('Predict'):

    val= scaler.transform([[val1,val2,val3,val4,val5]])
    prediction = model.predict([features])
    st.write(f"Predicted House Price: ${prediction[0]}")
    