# 🏦 Loan Approval Prediction System

This project was built as part of learning, machine learning and deploying models using Streamlit.

## 🧠 Problem Statement

Banks receive large number of loan applications, and checking them manually takes a lot of time and is error prone.
In this project, I built a uses Machine Learning model to predict whether a loan will be **Approved or Rejected** based on applicant details.

---

## 🚀 Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Preprocessing
- 🤖 Model Training & Comparison
- 📈 Feature Importance Analysis
- 🖥️ Streamlit Web App for predictions

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit

---

## 📊 Dataset

The dataset contains financial and personal details of loan applicants:

- Number of dependents
- Education
- Self-employed or not
- Income
- Loan amount
- Loan term
- CIBIL score
- Asset values

---

## 🔄 Project Workflow

1. Loaded and understood raw data
2. Data cleaning
3. Exploratory Data Analysis
4. Feature Encoding & Scaling
5. Model Training & Comparison
6. Final Model Selection and Saving
7. Built Streamlit website

---

## 🤖 Models Compared

- Logistic Regression
- Decision Tree
- Random Forest (Final Model used)

---

## 📈 Results & Insights

- CIBIL score is the most important feature
- Higher income and assets increase approval chances
- Random Forest gave the best performance

---

## 🖥️ How to Run This Project

### Clone repository

```bash
git clone https://github.com/vaishnavisingla222/loan_approval_ml.git
cd loan_approval_ml
```

### Create environment

```bash
python -m venv loan_env
source loan_env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run app

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

- Deploy app online 🌍
- Add database integration for storing user inputs
- Improve UI/UX of the Streamlit app

---

## 🙌 Author

**Vaishnavi Singla**

---

## 💬 Feedback

Feel free to explore the project, run it locally, and share your feedback.
Your suggestions and improvements are always welcome!

## ⭐ If you like this project, give it a star!
