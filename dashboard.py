import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pickle
import time

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1, h2, h3 {
    color: #123458;
}
.stButton>button {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(to right, #2a5298, #1e3c72);
    color: white;
}
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# Load model and scaler
model = pickle.load(open("model/loan_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

# Sidebar
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.title("🏦 Loan Predictor")
    st.markdown("---")
    st.info("""
    ### About
    This AI system predicts whether
    a loan application is likely
    to be approved or rejected
    based on applicant details.
    """)

    st.markdown("---")

    st.success("✅ Machine Learning Powered")
    st.success("📊 Interactive Dashboard")
    st.success("⚡ Real-Time Prediction")

# Main Content
st.title("🏦 Loan Approval Prediction System")

st.markdown("""
### Predict loan approval using Machine Learning 🤖

Check whether your loan application is likely to be approved based on:
- Financial Profile
- Credit Score
- Assets
- Income
- Loan Details
""")

st.divider()

# Applicant Details
st.subheader("👤 Applicant Details")
col1, col2 = st.columns(2)

with col1:
    dependent = st.number_input("👨‍👩‍👧 Dependents", min_value=0,step=1)
    education = st.selectbox("🎓 Education",["Select", "Graduate", "Not Graduate"])

with col2:
    selfEmployed = st.selectbox("💼 Self Employed",["Select", "Yes", "No"])

st.divider()

# Financial Details
st.subheader("💰 Financial Details")
income = st.number_input(
    "💵 Annual Income",
    min_value=0
)
col3, col4 = st.columns(2)
with col3:

    luxuryVal = st.number_input(
        "🚘 Luxury Assets Value",
        min_value=0
    )
    residentialVal = st.number_input(
        "🏠 Residential Assets Value",
        min_value=0
    )

with col4:
    bankVal = st.number_input(
        "🏦 Bank Assets Value",
        min_value=0
    )
    commercialVal = st.number_input(
        "🏢 Commercial Assets Value",
        min_value=0
    )

st.divider()

# Loan Details
st.subheader("📄 Loan Details")
col5, col6 = st.columns(2)
with col5:
    loanAmt = st.number_input(
        "💳 Loan Amount",
        min_value=0
    )
with col6:
    loanTerm = st.number_input(
        "📅 Loan Term (Months)",
        min_value=0
    )

st.divider()

# Credit Details
st.subheader("📊 Credit Details")

cibil = st.slider(
    "💳 CIBIL Score",
    300,
    900,
    650
)

# CIBIL Category
if cibil < 550:
    category = "Poor 😟"
    color = "red"

elif cibil < 650:
    category = "Average 😐"
    color = "orange"

elif cibil < 750:
    category = "Good 🙂"
    color = "blue"

else:
    category = "Excellent 😄"
    color = "green"

st.markdown(
    f"### CIBIL Category: :{color}[{category}]"
)

st.divider()

# Summary of Applicant
st.subheader("📌 Applicant Summary")

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "💵 Income",
    f"₹{income:,}"
)
m2.metric(
    "💳 Loan",
    f"₹{loanAmt:,}"
)
m3.metric(
    "📊 CIBIL",
    cibil
)
m4.metric(
    "👨‍👩‍👧 Dependents",
    dependent
)

st.divider()

if st.button("🔍 Predict Loan Status"):

    # Validation
    if education == "Select" or selfEmployed == "Select":
        st.warning("⚠️ Please fill all required fields")

    else:

        # Encode categorical values
        educationVal = 1 if education == "Graduate" else 0
        selfEmployedVal = 1 if selfEmployed == "Yes" else 0

        # Create dataframe
        input_data = pd.DataFrame([[
            dependent,
            educationVal,
            selfEmployedVal,
            income,
            loanAmt,
            loanTerm,
            cibil,
            residentialVal,
            commercialVal,
            luxuryVal,
            bankVal

        ]], columns=features)

        # Scale data
        inputScaled = scaler.transform(input_data)

        # Loading animation
        with st.spinner("🤖 AI is analyzing your profile..."):
            time.sleep(2)

            prediction = model.predict(inputScaled)[0]
            chances = model.predict_proba(inputScaled)[0]

        approval_probability = chances[1]
        loan_income_ratio = loanAmt / max(income, 1)

        st.divider()

        # Prediction Result
        st.subheader("🎯 Prediction Result")

        if prediction == 1:

            st.success(
                f"""
                ✅ Congratulations!

                Loan is likely to be APPROVED

                Confidence: {approval_probability*100:.2f}%
                """
            )

            if loan_income_ratio > 20:
                risk = "🔴 Approved by model, but financially VERY HIGH RISK"

            elif loan_income_ratio > 10:
                risk = "🟠 Approved by model, but HIGH RISK"

            else:
                risk = "🟢 Low Risk Applicant"

        else:

            st.error(
                f"""
                ❌ Loan may be REJECTED

                Confidence: {(1-approval_probability)*100:.2f}%
                """
            )

            risk = "🔴 High Risk Applicant"

        st.info(risk)

        st.divider()

        # Risk Analysis
        st.subheader("⚠️ Risk Analysis")

        if loan_income_ratio > 20:

            st.error(
                f"""
                Extremely High Risk

                Loan Amount is {loan_income_ratio:.1f}× Annual Income.
                """
            )

        elif loan_income_ratio > 10:

            st.warning(
                f"""
                High Risk

                Loan Amount is {loan_income_ratio:.1f}× Annual Income.
                """
            )

        elif loan_income_ratio > 5:

            st.info(
                f"""
                Moderate Risk

                Loan Amount is {loan_income_ratio:.1f}× Annual Income.
                """
            )

        else:

            st.success(
                f"""
                Healthy Loan-to-Income Ratio

                Ratio: {loan_income_ratio:.1f}×
                """
            )

        # Additional Risk Flags
        if dependent >= 5:
            st.warning(
                "👨‍👩‍👧 Large number of dependents may affect repayment capacity."
            )

        if cibil < 650:
            st.warning(
                "📉 Below-average CIBIL score may reduce approval chances."
            )

        if loanTerm > 60:
            st.warning(
                "📅 Long loan tenure may increase repayment burden."
            )

        st.divider()

        # Approval Probability Meter
        st.subheader("📈 Approval Probability Meter")

        figGauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=approval_probability * 100,
            title={'text': "Approval Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff4d4d"},
                    {'range': [40, 70], 'color': "#ffd633"},
                    {'range': [70, 100], 'color': "#66ff66"}
                ]
            }
        ))

        st.plotly_chart(
            figGauge,
            use_container_width=True
        )

        st.divider()

        # Asset Distribution
        st.subheader("📊 Applicant Asset Distribution")

        asset_data = pd.DataFrame({
            "Assets": [
                "Residential",
                "Commercial",
                "Luxury",
                "Bank"
            ],
            "Value": [
                residentialVal,
                commercialVal,
                luxuryVal,
                bankVal
            ]
        })

        figPie = px.pie(
            asset_data,
            names="Assets",
            values="Value",
            hole=0.4,
            title="Asset Distribution"
        )

        st.plotly_chart(
            figPie,
            use_container_width=True
        )

        st.divider()

        # Income vs Loan Amount
        st.subheader("💰 Income vs Loan Amount")

        comparison = pd.DataFrame({
            "Category": ["Income", "Loan Amount"],
            "Amount": [income, loanAmt]
        })

        figBar = px.bar(
            comparison,
            x="Category",
            y="Amount",
            text="Amount",
            title="Income vs Loan Amount"
        )

        st.plotly_chart(
            figBar,
            use_container_width=True
        )

        st.divider()

        # Financial Health Score
        st.subheader("🏆 Financial Health Score")

        healthScore = (
            (cibil / 900) * 50
            +
            min(income / (loanAmt + 1), 1) * 50
        )

        st.progress(int(healthScore))

        st.metric(
            "Overall Financial Health",
            f"{healthScore:.2f}/100"
        )

st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center'>

### 👨‍💻 Developed by Vaishnavi Singla

AI-Powered Loan Approval Prediction Dashboard 🚀

</div>
""", unsafe_allow_html=True)