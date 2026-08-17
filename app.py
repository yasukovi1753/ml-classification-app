"""
Breast Cancer Classification - Streamlit Web Application
Allows users to upload test data, select ML models,
view evaluation metrics, and see confusion matrices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Wisconsin - Classification Dashboard")
st.markdown("""
This application compares **5 Machine Learning models** on the 
**Breast Cancer Wisconsin (Diagnostic)** dataset.  
Upload your test CSV or use the built-in test split to explore model performance.
""")

st.divider()

# -------------------------------------------------------
# Helper: Train models on the fly (for deployment)
# -------------------------------------------------------
@st.cache_resource
def load_and_train_models():
    """Load dataset, train all models, return models + data."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
    }

    trained_models = {}
    for name, mdl in model_dict.items():
        mdl.fit(X_train_scaled, y_train)
        trained_models[name] = mdl

    return trained_models, scaler, X_test, y_test, data.feature_names, data.target_names


trained_models, scaler, default_X_test, default_y_test, feature_cols, target_labels = (
    load_and_train_models()
)

# -------------------------------------------------------
# Sidebar: Dataset Upload and Model Selection
# -------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test Data (CSV)", type=["csv"],
    help="CSV must contain the 30 feature columns and a 'target' column."
)

selected_model = st.sidebar.selectbox(
    "🤖 Select ML Model",
    list(trained_models.keys()),
    help="Choose which model's results to view in detail."
)

show_all = st.sidebar.checkbox("📊 Show All Models Comparison", value=True)

# -------------------------------------------------------
# Process uploaded data or use default test set
# -------------------------------------------------------
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {test_df.shape[0]} rows from uploaded file.")

    if "target" in test_df.columns:
        X_test_raw = test_df.drop(columns=["target"])
        y_test = test_df["target"]
    else:
        st.sidebar.error("CSV must contain a 'target' column!")
        st.stop()
else:
    X_test_raw = default_X_test
    y_test = default_y_test
    st.sidebar.info("Using built-in test split (114 samples).")

X_test_scaled = scaler.transform(X_test_raw)

# -------------------------------------------------------
# Compute metrics for all models
# -------------------------------------------------------
def compute_metrics(model, X_scaled, y_true):
    """Compute all 6 evaluation metrics for a model."""
    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_proba = y_pred

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


all_metrics = {}
for name, mdl in trained_models.items():
    all_metrics[name] = compute_metrics(mdl, X_test_scaled, y_test)

# -------------------------------------------------------
# Section 1: All Models Comparison Table
# -------------------------------------------------------
if show_all:
    st.header("📊 All Models - Comparison Table")

    comparison_df = pd.DataFrame(all_metrics).T
    comparison_df.index.name = "Model"

    # Highlight the best value in each column
    st.dataframe(
        comparison_df.style.highlight_max(axis=0, color="#90EE90"),
        use_container_width=True
    )

    # Bar chart comparing accuracies
    fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    bars = ax_acc.bar(comparison_df.index, comparison_df["Accuracy"], color=colors)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Model Accuracy Comparison")
    ax_acc.set_ylim(0.85, 1.0)
    for bar, val in zip(bars, comparison_df["Accuracy"]):
        ax_acc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", fontsize=9)
    plt.xticks(rotation=25, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_acc)

    # Best model callout
    best_name = comparison_df["Accuracy"].idxmax()
    best_acc = comparison_df.loc[best_name, "Accuracy"]
    st.success(f"**Best Model (by Accuracy):** {best_name} — {best_acc:.4f}")

    st.divider()

# -------------------------------------------------------
# Section 2: Selected Model - Detailed Results
# -------------------------------------------------------
st.header(f"🔍 Detailed Results — {selected_model}")

model = trained_models[selected_model]
y_pred = model.predict(X_test_scaled)
metrics = all_metrics[selected_model]

# Display metrics as cards
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
col2.metric("AUC", f"{metrics['AUC']:.4f}")
col3.metric("Precision", f"{metrics['Precision']:.4f}")
col4.metric("Recall", f"{metrics['Recall']:.4f}")
col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
col6.metric("MCC", f"{metrics['MCC']:.4f}")

st.divider()

# -------------------------------------------------------
# Section 3: Confusion Matrix
# -------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_labels, yticklabels=target_labels, ax=ax_cm)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title(f"Confusion Matrix — {selected_model}")
    plt.tight_layout()
    st.pyplot(fig_cm)

# -------------------------------------------------------
# Section 4: Classification Report
# -------------------------------------------------------
with col_right:
    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=target_labels, output_dict=True
    )
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

st.divider()

# -------------------------------------------------------
# Section 5: Dataset Information
# -------------------------------------------------------
with st.expander("📋 Dataset Information"):
    st.markdown("""
    **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
    **Source:** UCI Machine Learning Repository / sklearn  
    **Instances:** 569  
    **Features:** 30 (all numerical)  
    **Target:** Binary — Malignant (0) / Benign (1)  

    The features are computed from digitized images of fine needle 
    aspirate (FNA) of breast masses. They describe characteristics 
    of cell nuclei present in the image.

    **Feature Groups (10 real-valued features, each computed as mean, SE, and worst):**
    - Radius, Texture, Perimeter, Area, Smoothness
    - Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
    """)

    st.write("**Test Data Preview:**")
    preview_df = pd.DataFrame(X_test_raw, columns=feature_cols)
    preview_df["target"] = y_test.values
    st.dataframe(preview_df.head(10), use_container_width=True)

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "ML Assignment 2 | Breast Cancer Classification Dashboard"
    "</p>",
    unsafe_allow_html=True,
)
