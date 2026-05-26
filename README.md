# Fraud Detection using Machine Learning

A machine learning project to detect fraudulent transactions from an imbalanced financial dataset. Built using Python, Pandas, Scikit-learn, and SMOTE for handling class imbalance.

---

## Problem Statement

Credit card and financial fraud costs billions of dollars globally every year. The challenge with fraud detection is that fraudulent transactions are extremely rare compared to legitimate ones — making it a classic **imbalanced classification problem**. This project builds and evaluates ML models to automatically flag suspicious transactions.

---

## Dataset

- **Source:** [Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/waddahali/fraud-detection)
- **Type:** Synthetic tabular data
- **Size:** 7000 transactions
- **Challenge:** Highly imbalanced classes (very few fraud cases vs normal transactions)
- **Features:** Transaction details including amount, hour, device type, distance from home, and more

> ⚠️ The dataset is **NOT included** in this repo. Download `fraud.csv` from the Kaggle link above and place it inside the `data/` folder before running the notebook.

---

## Project Structure

```
fraud-detection/
│
├── fraud_detection.ipynb     ← Main analysis notebook
├── fraud_detector.py         ← Reusable OOP class
├── requirements.txt          ← Python dependencies
├── README.md                 ← You are here
│
├── data/                     ← Dataset folder (not uploaded)
│   └── fraud.csv             ← Download from Kaggle link above
│
└── plots/                    ← Saved visualizations (auto-created on run)
    ├── class_imbalance.png
    ├── correlation_heatmap.png
    ├── confusion_matrix_Logistic_Regression.png
    ├── confusion_matrix_Random_Forest.png
    └── feature_importance.png
```

---

## Approach

1. **Exploratory Data Analysis (EDA)** — understand the data, find patterns, visualize class imbalance
2. **Data Cleaning & Preprocessing** — handle missing values, encode categorical features
3. **Handle Class Imbalance** — apply SMOTE (Synthetic Minority Oversampling Technique) to balance fraud vs non-fraud
4. **Model Training** — train two models:
   - Logistic Regression (baseline)
   - Random Forest Classifier (improved model)
5. **Evaluation** — compare models using Precision, Recall, F1-Score, and Confusion Matrix

---

## Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 93%      | 91%       | 89%    | 90%      |
| Random Forest       | 97%      | 96%       | 95%    | 95%      |

---

## Key Findings

- The dataset is highly imbalanced — fraud cases make up a very small percentage of all transactions
- Without handling imbalance, models tend to always predict "not fraud" and still show high accuracy (misleading!)
- SMOTE significantly improved the model's ability to detect actual fraud cases
- Random Forest outperformed Logistic Regression across all metrics, especially Recall
- Top features influencing fraud prediction: `transaction_amount`, `distance_from_home`, `hour_of_day`, `network_quality`, `prev_transactions`

---

## Tech Stack

- **Python 3.x**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualizations
- **Scikit-learn** — ML models and evaluation
- **Imbalanced-learn** — SMOTE for class imbalance

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/rabiaabbasi/fraud-detection.git
cd fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place it in the data/ folder as fraud.csv

# 4. Open the notebook
jupyter notebook fraud_detection.ipynb
```

---

## About Me

I'm an aspiring AI Engineer learning Python, Machine Learning, and Data Science.
This is one of my first end-to-end ML projects — feedback welcome!

**Connect:** [LinkedIn](https://www.linkedin.com/in/rabia-abbasi/) | [GitHub](https://github.com/rabiaabbasi/)