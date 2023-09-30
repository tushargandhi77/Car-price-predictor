import streamlit as st
import numpy as np
import pickle
import pandas as pd

pipe = pickle.load(open('LinearRegressionModel.pkl','rb'))
df = pickle.load(open('car_data.pkl','rb'))

st.title("Car Price Prediction")

car_mode = st.selectbox('Car Model',df['name'].unique())

campany = st.selectbox('Company',df['company'].unique())

year = st.selectbox('Year',df['year'].unique())

km_driven = st.number_input('Kilometer driven')

fuel_type = st.selectbox('Fuel Type',df['fuel_type'].unique())

if st.button('Predict Price'):
    query = np.array([car_mode,campany,year,km_driven,fuel_type])
    query.reshape(1,5)
    st.title(str(pipe.predict(query)))