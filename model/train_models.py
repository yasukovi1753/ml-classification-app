"""
Breast Cancer Classification - Model Training Script
Trains 5 ML models on the Wisconsin Breast Cancer Dataset
and evaluates them using 6 metrics each.
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# -------------------------------------------------------
# 1. Load the Breast Cancer Wisconsin Dataset
# -------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"Dataset shape: {X.shape}")
print(f"Classes: {dict(zip([0, 1], data.target_names))}")
print(f"Class distribution:\n{y.value_counts()}\n")

# -------------------------------------------------------
# 2. Split data into training and testing sets
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples\n")

# -------------------------------------------------------
# 3. Feature Scaling using StandardScaler
# -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------
# 4. Save test data as CSV for Streamlit app
# -------------------------------------------------------
test_df = pd.DataFrame(X_test, columns=data.feature_names)
test_df["target"] = y_test.values
test_df.to_csv(os.path.join("..", "test_data.csv"), index=False)
print("Saved test_data.csv\n")

# -------------------------------------------------------
# 5. Define all classification models
# -------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Naive Bayes (Gaussian)": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
}

# -------------------------------------------------------
# 6. Train each model and compute evaluation metrics
# -------------------------------------------------------
results = {}

for name, model in models.items():
    print(f"Training: {name}...")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # For AUC, we need probability scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = y_pred

    # Calculate all 6 evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4),
    }

    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Precision:{prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC:      {mcc:.4f}")
    print()

# -------------------------------------------------------
# 7. Print Comparison Table
# -------------------------------------------------------
print("=" * 90)
print("MODEL COMPARISON TABLE")
print("=" * 90)
results_df = pd.DataFrame(results).T
print(results_df.to_string())
print()

# Find the best model based on accuracy
best_model_name = results_df["Accuracy"].idxmax()
print(f"Best Model (by Accuracy): {best_model_name} "
      f"({results_df.loc[best_model_name, 'Accuracy']:.4f})")

# -------------------------------------------------------
# 8. Save trained models and scaler
# -------------------------------------------------------
model_dir = os.path.dirname(os.path.abspath(__file__))

# Save the scaler
with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save each trained model
model_filenames = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

for name, model in models.items():
    filepath = os.path.join(model_dir, model_filenames[name])
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {model_filenames[name]}")

print("\nAll models saved successfully!")
print("\n--- Classification Reports ---\n")

# Print detailed classification reports
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred,
          target_names=data.target_names))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
