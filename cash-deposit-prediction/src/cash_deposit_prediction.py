"""
Cash Deposit Prediction Project
Author: Nikita Porwal

Description:
Predict deposit amount for 2017 using historical data.
Includes preprocessing, feature engineering, and model comparison.
"""

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# ===============================
# LOAD DATA
# ===============================
def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train["Sample"] = "train"
    test["Sample"] = "test"

    data = pd.concat([train, test], axis=0, sort=False)
    return data, train, test

# ===============================
# DATA CLEANING
# ===============================
def clean_data(data):
    # Convert date
    data['date_of_establishment'] = pd.to_datetime(
        data['date_of_establishment'], errors='coerce'
    )

    # Fill missing values
    deposit_cols = [
        'deposit_amount_2011', 'deposit_amount_2012',
        'deposit_amount_2013', 'deposit_amount_2014',
        'deposit_amount_2015', 'deposit_amount_2016'
    ]

    for col in deposit_cols:
        data[col].fillna(data[col].mean(), inplace=True)

    data['date_of_establishment'].fillna(pd.Timestamp("2000-01-01"), inplace=True)

    return data

# ===============================
# FEATURE ENGINEERING
# ===============================
def feature_engineering(data):
    data["Year_Established"] = data["date_of_establishment"].dt.year

    # Drop unnecessary columns
    for col in ["id", "date_of_establishment"]:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    return data

# ===============================
# ENCODING
# ===============================
def encode_data(data):
    categorical_cols = ['location.Code', 'location', 'loc.details', 'state', 'Sample']

    for col in categorical_cols:
        if col in data.columns:
            lbl = LabelEncoder()
            data[col] = lbl.fit_transform(data[col].astype(str))

    data = pd.get_dummies(data)
    return data

# ===============================
# SPLIT DATA
# ===============================
def split_data(data, original_train):
    train_data = data[data["Sample"] == 1].copy()
    test_data = data[data["Sample"] == 0].copy()

    train_data.drop(columns=["Sample"], inplace=True)
    test_data.drop(columns=["Sample"], inplace=True)

    train_data["deposit_amount_2017"] = original_train["deposit_amount_2017"].values

    X = train_data.drop(columns=["deposit_amount_2017"])
    y = train_data["deposit_amount_2017"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# ===============================
# TRAIN MODELS
# ===============================
def train_models(X_train, X_val, y_train, y_val):
    results = {}

    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=5),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=0),
        "KNN": KNeighborsRegressor(n_neighbors=5)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        score = model.score(X_val, y_val)

        results[name] = {"RMSE": rmse, "Score": score}

    # Grid Search (Ridge)
    params = {'alpha': np.linspace(1, 10, 50)}
    ridge = GridSearchCV(Ridge(), param_grid=params, cv=5, scoring='neg_mean_absolute_error')
    ridge.fit(X_train, y_train)

    preds = ridge.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    score = ridge.score(X_val, y_val)

    results["Ridge (GridSearch)"] = {"RMSE": rmse, "Score": score}

    return results

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    print(" Loading Data...")
    data, train, test = load_data()

    print(" Cleaning Data...")
    data = clean_data(data)

    print(" Feature Engineering...")
    data = feature_engineering(data)

    print(" Encoding...")
    data = encode_data(data)

    print(" Splitting Data...")
    X_train, X_val, y_train, y_val = split_data(data, train)

    print(" Training Models...")
    results = train_models(X_train, X_val, y_train, y_val)

    print("\n Model Performance:")
    for model, metrics in results.items():
        print(f"{model}: RMSE = {metrics['RMSE']:.2f}, Score = {metrics['Score']:.4f}")


if __name__ == "__main__":
    main()
