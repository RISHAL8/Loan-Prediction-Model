import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/data.csv")

# Drop ID column if present
if "Loan_ID" in df.columns:
    df.drop(["Loan_ID"], axis=1, inplace=True)

# Feature engineering: use TotalIncome only
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df.drop(["ApplicantIncome", "CoapplicantIncome"], axis=1, inplace=True)

# Target mapping
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Basic cleanup
df = df.dropna(subset=["Loan_Status"])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"].astype(int)

# Identify columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("cat", cat_pipe, cat_cols),
    ("num", num_pipe, num_cols)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save SHAP background sample
os.makedirs("models", exist_ok=True)
background = X_train.sample(min(100, len(X_train)), random_state=42)
joblib.dump(background, "models/shap_background.pkl")

# Class imbalance handling for XGBoost
pos = max(int((y_train == 1).sum()), 1)
neg = max(int((y_train == 0).sum()), 1)
scale_pos_weight = neg / pos

# Models
xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    ))
])

bagging_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5, class_weight="balanced"),
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

# Train
xgb_model.fit(X_train, y_train)
bagging_model.fit(X_train, y_train)

# Evaluate
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob))
    }

xgb_metrics = evaluate(xgb_model, X_test, y_test)
bagging_metrics = evaluate(bagging_model, X_test, y_test)

# Save models
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(bagging_model, "models/bagging_model.pkl")

# Save metrics
metrics = {
    "xgboost": xgb_metrics,
    "bagging": bagging_metrics
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training completed")
print("XGBoost metrics:", xgb_metrics)
print("Bagging metrics:", bagging_metrics)
print("✅ Saved: models/xgb_model.pkl, models/bagging_model.pkl, models/shap_background.pkl, models/metrics.json")