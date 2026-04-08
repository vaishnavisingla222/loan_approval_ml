import streamlit as st
import pandas as pd
import pickle

# Load model files
model = pickle.load(open("model/loan_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

st.set_page_config(page_title="Loan Approval App", layout="centered")

# Title
st.title("Loan Approval Prediction System")
st.markdown("### Check whether your loan will be approved or rejected based on your financial and personal details.")

st.divider()

# Add Personal Details
st.subheader("Applicant Details")
col1, col2 = st.columns(2)
with col1:
    dependent = st.number_input("Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    selfEmployed = st.selectbox("Self Employed", ["Yes", "No"])

st.divider()

# Add Financial Details
st.subheader("Financial Details")
income = st.number_input("Annual Income", min_value=0)
col3, col4 = st.columns(2)

with col3:
    luxuryVal = st.number_input("Luxury Assets", min_value=0)
    residentialVal = st.number_input("Residential Assets", min_value=0)

with col4:
    bankVal = st.number_input("Bank Assets", min_value=0)
    commercialVal = st.number_input("Commercial Assets", min_value=0)

st.divider()

# Add Loan Details
st.subheader("Loan Details")
col5, col6 = st.columns(2)
with col5:
    loanAmt = st.number_input("Loan Amount", min_value=0)
with col6:
    loanTerm = st.number_input("Loan Term (months)", min_value=0)

st.divider()

# Add Credit Details
st.subheader("Credit Details")
cibil = st.slider("CIBIL Score", 300, 900, 650)

# CIBIL category
if cibil < 550:
    category = "Poor"
elif cibil < 650:
    category = "Average"
elif cibil < 750:
    category = "Good"
else:
    category = "Excellent"

st.info(f" CIBIL Category: {category}")

st.divider()

# Preprocess categorical inputs
if education == "Graduate":
    education = 1 
else:
    education = 0

if selfEmployed == "Yes":
    selfEmployed = 1
else:
    selfEmployed = 0

# Prediction Button
if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([[
        dependent, education, selfEmployed,
        income,residentialVal,commercialVal,luxuryVal, bankVal,
        loanAmt, loanTerm,
        cibil
    ]], columns=features)

    inputScaled = scaler.transform(input_data)
    prediction = model.predict(inputScaled)[0]
    
    chances = model.predict_proba(inputScaled)[0]
    if prediction == 1:
        probability = chances[1]
    else:
        probability = chances[0]

    st.divider()

    # Results
    if prediction == 1:
        st.success(f"Congratulations! Loan can be Approved ({probability*100:.2f}% confidence)")
        st.markdown("Strong financial profile and credit score")
    else:
        st.error(f"Sorry, your loan can be rejected ({probability*100:.2f}% confidence)")
        st.markdown("Kindly improve CIBIL score or income to increase chances")
