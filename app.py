#!/usr/bin/env python
# coding: utf-8

# In[3]:



import streamlit as st
import pickle
import numpy as np
import json
from streamlit_lottie import st_lottie

# Load the trained model
model = pickle.load(open('loan_model.pkl', 'rb'))

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter the applicant's details below to predict loan approval:")

# Load Lottie JSON from local files
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

success_anim = load_lottiefile("success_animation.json")
fail_anim = load_lottiefile("failure_animation.json")

with st.sidebar:
    st.title("📊 About This App")
    st.markdown("""
    This app predicts whether a loan will be **approved** or **rejected** based on applicant data.

    **Built with**:
    - Python
    - Numpy
    - Pandas
    - Streamlit
    - Scikit-learn
    - Lottie Animations

    Upload your model: `loan_model.pkl`  
    Add animations: `success.json`, `fail.json`, `header.json`

    Developed by **Prajukta Mandal** 💡
""")
    
               

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [360, 120, 180, 300, 240, 84, 60, 12])
credit_history = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encoding function
def encode_inputs():
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)
    education_val = 0 if education == "Graduate" else 1
    self_emp_val = 1 if self_employed == "Yes" else 0
    prop_val = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    return np.array([[gender_val, married_val, dependents_val, education_val,
                      self_emp_val, applicant_income, coapplicant_income,
                      loan_amount, loan_term, credit_history, prop_val]])




# Trigger prediction only when real button is clicked
    if st.button("🚀 Predict Loan Status"):
        features = encode_inputs()
        prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
        st_lottie(success_anim, speed=1, height=300)
    else:
        st.error("❌ Loan Not Approved.")
        st_lottie(fail_anim, speed=1, height=300)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Made with ❤️ by Streamlit</p>", unsafe_allow_html=True)


# In[ ]:




