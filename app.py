import streamlit as st
import pandas as pd
import pickle

# Load model files
model = pickle.load(open("model/loan_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

st.set_page_config(page_title="Loan Approval App", layout="centered")

# Title
st.title("🏦 Loan Approval Prediction System")
st.markdown("### 📊 Check whether your loan will be approved")

st.divider()

# SECTION 1: PERSONAL DETAILS
st.subheader("👤 Applicant Details")
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    loan_term = st.number_input("Loan Term (months)", min_value=0)

st.divider()

# SECTION 2: FINANCIAL DETAILS
st.subheader("💰 Financial Details")
col3, col4 = st.columns(2)

with col3:
    income_annum = st.number_input("Annual Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)

with col4:
    bank_asset_value = st.number_input("Bank Assets", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets", min_value=0)

residential_assets_value = st.number_input("Residential Assets", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets", min_value=0)

st.divider()

# SECTION 3: CREDIT DETAILS
st.subheader("📈 Credit Details")
cibil_score = st.slider("CIBIL Score", 300, 900, 650)

# CIBIL category
if cibil_score < 550:
    category = "Poor"
elif cibil_score < 650:
    category = "Average"
elif cibil_score < 750:
    category = "Good"
else:
    category = "Excellent"

st.info(f"💡 CIBIL Category: {category}")

st.divider()

# PREPROCESS INPUT
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# PREDICTION BUTTON
if st.button("🚀 Predict Loan Status"):

    input_data = pd.DataFrame([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]], columns=features)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]
    st.divider()

    # RESULT DISPLAY
    if prediction == 1:
        st.success(f"✅ Loan Approved ({probability*100:.2f}% confidence)")
        st.markdown("💡 Strong financial profile and credit score")
    else:
        st.error(f"❌ Loan Rejected ({probability*100:.2f}% confidence)")
        st.markdown("⚠️ Improve CIBIL score or income to increase chances")