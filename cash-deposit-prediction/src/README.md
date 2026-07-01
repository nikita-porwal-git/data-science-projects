# Cash Deposit Prediction (Machine Learning Project)

## Overview
Predict cash deposit amount for 2017 using historical financial data.

## Problem Statement
Build a regression model to forecast deposit amounts based on past years' data.

## Approach
- Data cleaning and preprocessing
- Feature engineering (Year Established)
- Handling missing values and outliers
- Model training and evaluation

## Models Used
- Decision Tree Regressor
- Linear Regression
- Random Forest Regressor (Best)
- KNN Regressor
- Ridge Regression (GridSearchCV)

## Results
Random Forest achieved the best performance with lowest RMSE.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Project Structure

cash-deposit-prediction/
│
├── src/
│   └── cash_deposit_prediction.py
├── data/
│   ├── train.csv
│   └── test.csv
├── requirements.txt
├── README.md

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Run: python src/cash_deposit_prediction.py

## Author
Nikita Porwal

