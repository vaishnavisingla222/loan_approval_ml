# 🏦 Loan Approval Prediction System

An Supervised Machine Learning project that predicts whether a loan application will be approved or rejected based on applicant financial and personal details.

Built using Python, Scikit-learn, and Streamlit with an interactive dashboard for real-time predictions.

---

# 🌐 Live Demo

🔗 [Open Streamlit App](https://loanapprovalpridiction-h25gkuw8flzcb8myaubv8q.streamlit.app/)

---

# 🧠 Problem Statement

Banks receive a large number of loan applications daily.
Manual verification is time-consuming and may lead to inconsistent decisions.

This project uses Machine Learning to automate and improve the loan approval prediction process using applicant financial information.

---

# ⚙️ Tech Stack

| Technology           | Purpose            |
| -------------------- | ------------------ |
| Python               | Core Programming   |
| Pandas & NumPy       | Data Processing    |
| Scikit-learn         | Machine Learning   |
| Matplotlib & Seaborn | Data Visualization |
| Streamlit            | Web App Deployment |

---

# 📊 Dataset Information

The dataset contains applicant financial and personal information including:

| Feature                |
| ---------------------- |
| Dependents             |
| Education              |
| Self Employment Status |
| Annual Income          |
| Loan Amount            |
| Loan Term              |
| CIBIL Score            |
| Residential Assets     |
| Commercial Assets      |
| Luxury Assets          |
| Bank Assets            |

---

# 🔄 Machine Learning Workflow

```text
Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
EDA & Visualization
      ↓
Feature Encoding & Scaling
      ↓
Model Training & Comparison
      ↓
Hyperparameter Tuning
      ↓
Final Model Selection
      ↓
Streamlit Dashboard Deployment
```

---

# 🤖 Models Compared

| Model               | Performance |
| ------------------- | ----------- |
| Logistic Regression | Good        |
| Decision Tree       | Better      |
| Random Forest       | Best ✅     |

---

# 📈 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 97.78% |
| Precision | 97.78% |
| Recall    | 98.69% |
| F1-Score  | 98.24% |

---

# 📌 Key Insights

- 📈 Higher CIBIL scores significantly improve approval chances
- 💰 Higher income and asset values positively influence predictions
- 🏦 Random Forest outperformed other classification models

---

# 🖥️ Dashboard Features

✅ Real-time loan prediction
✅ Interactive financial analysis graphs
✅ Loan approval confidence score
✅ Asset distribution visualization
✅ Income vs Loan comparison chart

---

# 🚀 How to Run Locally

## Clone Repository

```bash
git clone https://github.com/vaishnavisingla222/loan_approval_ml.git
cd loan_approval_ml
```

## Create Virtual Environment

```bash
python -m venv loan_env
source loan_env/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```text
loan_approval_ml/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── loan_model.pkl
│   ├── scaler.pkl
│   └── features.pkl
```

---

# 🎯 Future Improvements

- 🌍 Deploy dashboard publicly
- 🗄️ Add database integration
- 🎨 Improve UI/UX
- 📊 Add advanced analytics dashboard

---

# 👩‍💻 Author

**Vaishnavi Singla**

🔗 GitHub: https://github.com/vaishnavisingla222

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
