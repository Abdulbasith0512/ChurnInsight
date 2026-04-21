# 🚀 Customer Churn Intelligence System (ML + SQL + Dashboard)

[![Live App](https://img.shields.io/badge/🚀%20Live%20Dashboard-Open%20Now-success)](https://churninsight-hvwtgz7vj9tgt5pyevpxvn.streamlit.app/)

## 🌐 Live Demo

👉 **[Click here to explore the Streamlit App](https://churninsight-hvwtgz7vj9tgt5pyevpxvn.streamlit.app/)**

🔗 Experience real-time churn prediction with interactive analytics, KPIs, and business insights.

---

## 📌 Overview

Customer churn is a critical business problem that directly impacts revenue.
This project builds a complete **churn intelligence system** that not only predicts churn but also explains *why customers leave* and provides data-driven recommendations.


---

## 🎯 Objectives

* Predict customer churn using machine learning models
* Identify key factors driving churn
* Perform hypothesis testing on business assumptions
* Enable data-driven decision-making through dashboards

---

## 🧠 Key Features

### ✔ Data Analysis & EDA

* Churn distribution and customer segmentation
* Behavioral analysis across contract types, tenure, and billing
* Visual insights using Matplotlib & Seaborn

---

### ✔ Hypothesis Testing

* Validated business assumptions using statistical tests
* Example:

  * Do higher monthly charges increase churn?
  * Does contract type influence retention?

---

### ✔ Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest

📊 Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

### ✔ SQL Integration

* Stored dataset in SQL database
* Performed analytical queries:

  * Churn rate calculation
  * Churn by contract type
  * Revenue impact analysis

---

### ✔ Interactive Dashboard

* Built using **Streamlit**
* Features:

  * Churn rate KPI
  * Dynamic filters
  * Visual analytics
  * Real-time churn prediction

---

## 📂 Project Structure

```id="a7gk3x"
Customer-Churn-Intelligence-System/
│── data/
│     ├── raw/
│     ├── processed/
│
│── notebooks/
│     ├── 01_data_understanding.ipynb
│     ├── 02_eda.ipynb
│     ├── 03_feature_engineering.ipynb
│     ├── 04_modeling.ipynb
│     ├── 05_evaluation.ipynb
│
│── src/
│     ├── preprocessing.py
│     ├── feature_engineering.py
│     ├── model.py
│     ├── evaluation.py
│
│── sql/
│     └── churn_analysis.sql
│
│── dashboard/
│     └── app.py
│
│── requirements.txt
│── README.md
```

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash id="1x9c2d"
git clone https://github.com/your-username/customer-churn-intelligence-system.git
cd customer-churn-intelligence-system
```

---

### 2️⃣ Install Dependencies

```bash id="7azv4n"
pip install -r requirements.txt
```

---

### 3️⃣ Run Jupyter Notebooks

```bash id="l8n3qs"
jupyter notebook
```

---

### 4️⃣ Run Dashboard

```bash id="9r1k2m"
streamlit run dashboard/app.py
```

---

## 📊 Dataset Description

| Feature         | Description               |
| --------------- | ------------------------- |
| tenure          | Customer duration         |
| monthly_charges | Monthly billing amount    |
| contract_type   | Contract category         |
| payment_method  | Mode of payment           |
| churn           | 0 = Retained, 1 = Churned |

---

## 📈 Key Insights

* Customers with **month-to-month contracts** show higher churn rates
* **Higher monthly charges** are correlated with increased churn
* Long-term customers exhibit **lower churn probability**
* Contract type plays a significant role in retention

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn, Plotly
* Scikit-learn
* SQL (SQLite/MySQL)
* Streamlit

---

## 🔮 Future Improvements

* Deploy model using Flask/FastAPI
* Use advanced models (XGBoost, LightGBM)
* Real-time churn prediction pipeline
* Integration with live business data

---

## 💼 Business Impact

This system helps businesses:

* Identify high-risk customers
* Reduce churn through targeted strategies
* Improve customer retention and revenue

---

## 👩‍💻 Author

**Abdul Basith**

---

⭐ If you found this project useful, consider giving it a star!
