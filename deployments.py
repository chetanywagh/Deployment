# -*- coding: utf-8 -*-
"""
Created on Tue May 27 18:21:47 2025

@author: Lenovo
"""

import streamlit as st
import pandas as pd
import pickle

with open(r'Titanic_mode.pkl', 'rb') as file:
    titanic_logr_model = pickle.load(file)
    
def prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    

    features = pd.DataFrame([{
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }])
    preds = titanic_logr_model.predict(features)[0]
    return preds

st.title("Titanic Survival Prediction 🚢")

Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare Paid", min_value=0.0)
Embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict Now :mag_right:"):
    pred = prediction(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    st.write(f"Prediction: {'😀 Survived' if pred == 1 else  '💀 Did not survive'}")
