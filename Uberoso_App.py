# Required libraries
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
import streamlit as st

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Defining Prediction Function

def predict_use(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]




# Loading Model

model = load_model('final_lgbm_12262022')


# Writing the title of the app a a brief description.

st.write("""
# Uberoso Service Estimation (Prediction) App

##### Author: Alexis Vera, MPH, DBA

Use this app to estimate/predict the total rides in a day with specific characteristics.


""")

st.write('Fill-in the information below and then run the app to obtain a predicted value of total rides.')

# Making Sliders and Feature Variables

temp = st.sidebar.number_input(label = 'Temperature (F)', value=75, step=1, help='Enter the temperature in Farenheit.')

Month = st.sidebar.selectbox(label = 'Month', options=[1,2,3,4,5,6,7,8,9,10,11,12], help='Select the month.')

Hour = st.sidebar.selectbox(label = 'Hour', options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                                    help='Select the time of the day. The hour should be entered using 24-hr format.')

Season = st.sidebar.selectbox(label = 'Season', options=['Fall','Spring','Summer','Winter'], help='Select the Season.')

Day = st.sidebar.selectbox(label = 'Day of the Week', options=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], help='Select the day of the week.')


# Mapping feature labels with input values

features = {'temp':temp,
            'Month':Month,
            'Hour':Hour,
            'Season':Season,
            'Day':Day
}


# Converting Features into DataFrame

features_df  = pd.DataFrame([features])

st.table(features_df)


# Predicting Uberoso Use

if st.button('Predict'):
    
    prediction = predict_use(model, features_df)
    
    st.write('Based on input data, the predicted Uberoso Use is '+ prediction)