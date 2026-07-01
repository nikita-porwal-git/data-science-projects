
# 🏦 Loan Default Prediction (Good vs Bad Customers)

## 📊 Overview
This project predicts whether a customer is likely to **default on a loan** (Bad Customer) or **repay successfully** (Good Customer).

It demonstrates an **end-to-end machine learning classification pipeline**, including preprocessing, handling imbalanced data, and model optimisation.

---

## 🎯 Problem Statement
Financial institutions need to minimise risk while granting loans.  

The objective is to:
- Identify **high-risk (defaulting) customers**
- Support **data-driven lending decisions**

---

## ⚙️ Project Workflow

### 🔹 Data Preprocessing
- Checked for missing values  
- Removed irrelevant columns (e.g., Customer ID)  
- Converted categorical variables into numerical format  
- Applied one-hot encoding  

---

### 🔹 Feature Engineering & Cleaning
- Removed multicollinearity between variables  
- Scaled features using StandardScaler  
- Cleaned and structured dataset for modelling  

---

### 🔹 Handling Imbalanced Data
The dataset had fewer default cases (imbalance issue).

✅ Applied **SMOTE (Synthetic Minority Oversampling Technique)**:
- Balanced minority class  
- Improved recall for default prediction  

---

## 🤖 Models Implemented

### ✅ Logistic Regression (Baseline)
- Initial model trained on original dataset  

### ✅ Logistic Regression with SMOTE
- Improved performance on minority class  

### ✅ Logistic Regression with Hyperparameter Tuning
- GridSearchCV used for:
  - Regularisation (C)
  - Penalty (L1, L2)
  - Class weights  

---

## 📈 Model Evaluation

Metrics used:
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

### ✅ Key Results:
- Improved recall after SMOTE  
- Further performance enhancement after model tuning  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**
  - pandas, numpy  
  - scikit-learn  
  - imbalanced-learn (SMOTE)  

---

## 📂 Project Structure
