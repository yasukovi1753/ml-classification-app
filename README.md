# Breast Cancer Wisconsin - Classification using Machine Learning

## a. Problem Statement

The goal of this project is to classify breast tumors as **malignant** or **benign** based on features extracted from digitized images of fine needle aspirate (FNA) of breast masses. Early and accurate diagnosis of breast cancer is critical for patient survival, and machine learning models can assist medical professionals in making faster, data-driven decisions. This project implements and compares 5 different ML classification algorithms to identify the most effective model for this task.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Dataset Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository / sklearn built-in |
| **Total Instances** | 569 |
| **Number of Features** | 30 (all numerical, real-valued) |
| **Target Variable** | Diagnosis — Malignant (0) or Benign (1) |
| **Class Distribution** | Malignant: 212 (37.3%), Benign: 357 (62.7%) |
| **Train/Test Split** | 80/20 stratified split (455 train, 114 test) |

**Feature Description:**
The 30 features are computed from digitized images of FNA of breast masses. They describe characteristics of cell nuclei present in the image. For each of 10 real-valued properties, the mean, standard error (SE), and worst (largest) values are recorded:

1. **Radius** — mean distance from center to perimeter points
2. **Texture** — standard deviation of grayscale values
3. **Perimeter** — perimeter of the cell nucleus
4. **Area** — area of the cell nucleus
5. **Smoothness** — local variation in radius lengths
6. **Compactness** — (perimeter^2 / area) - 1.0
7. **Concavity** — severity of concave portions of the contour
8. **Concave Points** — number of concave portions of the contour
9. **Symmetry** — symmetry of the cell nucleus
10. **Fractal Dimension** — coastline approximation - 1

This gives 30 features total (10 properties x 3 statistics each: mean, SE, worst).

---

## c. GitHub Repository Link

**Repository:** [Your GitHub Repository Link Here]

**Repository Structure:**
```
ml-classification-app/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test dataset (114 samples)
└── model/
    ├── train_models.py     # Model training script
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

---

## d. Models Used

### Model Comparison Table

All 5 models were trained on the same 80/20 stratified split with StandardScaler normalization. The following 6 evaluation metrics were calculated on the test set (114 samples):

| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **K-Nearest Neighbors** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes (Gaussian)** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

---

### Model Observations

| **ML Model Name** | **Observation about model performance** |
|---|---|
| **Logistic Regression** | Delivered the highest accuracy (98.25%) and best MCC (0.9623) among all models. With only 1 false positive and 1 false negative, it demonstrates that a simple linear boundary is highly effective for this dataset. The high AUC of 0.9954 confirms strong discriminative ability across all thresholds. This suggests the classes are nearly linearly separable in the scaled feature space. |
| **Decision Tree** | Recorded the lowest accuracy (92.11%) and AUC (0.9163), with 9 misclassifications total. The model is prone to overfitting on training data and struggles with generalization despite using max_depth=5 as a regularization measure. Its recall of 0.9167 indicates it misses some benign cases. Decision boundaries based on axis-aligned splits are less effective for this dataset's feature interactions. |
| **K-Nearest Neighbors** | Achieved 95.61% accuracy with balanced precision (0.9589) and recall (0.9722). KNN benefits significantly from StandardScaler normalization since it relies on distance calculations. With k=5, it captures local patterns effectively. The AUC of 0.9788 shows good ranking capability. Performance is competitive with Random Forest, indicating that local neighborhood information is useful for this classification task. |
| **Naive Bayes (Gaussian)** | Achieved 92.98% accuracy, slightly above Decision Tree. Despite assuming feature independence (which does not hold for this dataset since mean, SE, and worst features are correlated), it performs reasonably well. The high AUC of 0.9868 shows that its probability estimates capture class separation effectively even though hard predictions have more errors. This aligns with the known robustness of Naive Bayes to violated independence assumptions. |
| **Random Forest (Ensemble)** | Matched KNN in accuracy (95.61%) and achieved the second-highest AUC (0.9939). As an ensemble of 100 decision trees, it overcomes the single Decision Tree's overfitting problem through bagging and feature randomization. The near-perfect AUC demonstrates excellent ranking ability. However, its hard classification accuracy does not surpass Logistic Regression, suggesting that for this relatively clean, linearly-separable dataset, ensemble complexity does not always translate to better predictions. |
| **Overall Winner for this dataset?** | **Logistic Regression** is the clear winner with the highest Accuracy (0.9825), F1 (0.9861), and MCC (0.9623). Its superior performance indicates that the breast cancer features, after scaling, form a nearly linearly separable space. Logistic Regression also offers the advantages of interpretability (feature coefficients can indicate which measurements matter most) and computational efficiency. For clinical deployment, its combination of high accuracy, strong AUC, and model simplicity makes it the most suitable choice. |

---

## Streamlit Application

**Deployed App Link:** [Your Streamlit App URL Here]

### App Features:
1. **CSV Upload** — Upload custom test data in CSV format
2. **Model Selection** — Dropdown to select any of the 5 ML models
3. **Evaluation Metrics** — Displays Accuracy, AUC, Precision, Recall, F1, and MCC
4. **Confusion Matrix & Classification Report** — Visual heatmap and detailed per-class metrics

---

## How to Run Locally

```bash
# Clone the repository
git clone <your-repo-url>
cd ml-classification-app

# Install dependencies
pip install -r requirements.txt

# Train models (generates .pkl files and test_data.csv)
cd model
python train_models.py
cd ..

# Run the Streamlit app
streamlit run app.py
```

---

## Technologies Used

- **Python 3.10+**
- **scikit-learn** — ML model training and evaluation
- **Streamlit** — Interactive web application
- **Pandas & NumPy** — Data processing
- **Matplotlib & Seaborn** — Visualizations
