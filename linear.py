import streamlit as st
import joblib # pip install joblib
from sklearn.preprocessing import StandardScaler
st.header("Economic Index finder")

val1= st.text_input("Interest Rate")
val2= st.text_input("Unemployment Rate")


import pandas as pd
df= pd.read_csv('economic_index.csv')
df.drop(labels=['Unnamed: 0','year','month'], axis=1, inplace=True)
scaler= StandardScaler()
scaler.fit(df.loc[:,['interest_rate','unemployment_rate']])




if st.button("Check"):
    val= scaler.transform([[val1,val2]])
    st.write(val)
    model= joblib.load('ec.pkl')
    answer = model.predict(val)
    st.subheader(f"Result is {answer[0]}")
    