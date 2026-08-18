# Breast Cancer Wisconsin - Classification using Machine Learning

## a. Problem Statement

Breast cancer remains one of the leading causes of mortality among women worldwide. Accurate classification of tumors as malignant or benign at an early stage can significantly improve treatment outcomes. In this project, the Breast Cancer Wisconsin (Diagnostic) dataset is used to build and compare five different machine learning classification models. The objective is to evaluate which classifier performs the best on this medical diagnosis task using six standard evaluation metrics.

---

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository (also available as a built-in dataset in sklearn)

| Property | Details |
|----------|---------|
| Total Instances | 569 |
| Number of Features | 30 (all numerical) |
| Target Variable | Diagnosis — Malignant (0) or Benign (1) |
| Class Distribution | Malignant: 212 (37.3%), Benign: 357 (62.7%) |
| Train/Test Split | 80/20 stratified (455 train, 114 test) |
| Preprocessing | StandardScaler applied for feature normalization |

The dataset contains 30 real-valued features computed from digitized images of fine needle aspirate (FNA) of breast masses. These features describe various properties of the cell nuclei observed in the images. There are 10 base measurements — radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — and for each, three values are recorded: the mean, standard error, and the worst (i.e., largest) value. This gives a total of 30 features.

---

## c. GitHub Repository Link

**Repository:** https://github.com/yasukovi1753/ml-classification-app

**Streamlit App:** https://ml-classification-app-mmagck28cetkq8tyu6gijh.streamlit.app

Repository structure:

```
ml-classification-app/
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── test_data.csv                    # Test dataset (114 samples)
└── model/
    ├── ML_Assignment2_Models.ipynb  # Jupyter Notebook (model training + evaluation)
    ├── train_models.py              # Python script version
    ├── logistic_regression.pkl      # Saved trained models
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

---

## d. Models Used

Five classification models were trained on the same training set (455 samples) using StandardScaler for normalization. All six evaluation metrics were computed on the test set (114 samples).

### Comparison Table

| **ML Model Name** | **Accuracy** | **AUC** | **Precision** | **Recall** | **F1** | **MCC** |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **kNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest (Ensemble)** | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations

| **ML Model Name** | **Observation about model performance** |
|---|---|
| **Logistic Regression** | This model performed the best overall with 98.25% accuracy. It only misclassified 2 samples out of 114 — one false positive and one false negative. The MCC of 0.9623 is the highest among all models, which shows it handles both classes well despite the class imbalance. The strong performance makes sense because after scaling, the features seem to create a clear separation between malignant and benign cases, so a linear decision boundary works really well here. |
| **Decision Tree** | Decision Tree gave the weakest results with 92.11% accuracy and the lowest AUC (0.9163). It misclassified 9 test samples. Even with max_depth set to 5 to prevent overfitting, it still struggled compared to other models. The main issue is that Decision Tree makes axis-aligned splits, and this dataset has correlated features (like mean and worst values of the same property), so a single tree cannot capture those relationships as effectively. |
| **kNN** | KNN achieved 95.61% accuracy, performing on par with Random Forest. Since KNN works based on distance between data points, the StandardScaler normalization was crucial here — without it, features with larger ranges like area would dominate the distance calculation. With k=5, the model balances between being too sensitive to noise (low k) and losing local patterns (high k). Its recall of 0.9722 means it correctly identified most benign cases. |
| **Naive Bayes** | Naive Bayes scored 92.98% accuracy. The Gaussian variant was used since all 30 features are continuous. Interestingly, even though the independence assumption is clearly violated here (mean radius, SE of radius, and worst radius are obviously correlated), the model still performs reasonably. What stood out is its AUC of 0.9868 — much higher than Decision Tree — which means the probability scores it outputs rank samples well, even if the final hard predictions are not as accurate. |
| **Random Forest (Ensemble)** | Random Forest matched KNN in accuracy (95.61%) and had the second-highest AUC at 0.9939. Being an ensemble of 100 trees, it avoids the overfitting problem seen with a single Decision Tree. Each tree is trained on a random subset of features and samples, so the combined prediction is much more stable. However, even with this added complexity, it could not beat Logistic Regression on accuracy, which suggests that adding more model complexity is not always beneficial when the data is already well-separable. |
| **Overall Winner for this dataset?** | **Logistic Regression** wins on this dataset. It has the best Accuracy (0.9825), F1 (0.9861), and MCC (0.9623). Random Forest comes close in AUC (0.9939 vs 0.9954) but falls short on actual classification accuracy. The reason Logistic Regression works so well is that the breast cancer features, once normalized, form two groups that can be separated almost perfectly by a straight line (linear boundary). On top of performance, Logistic Regression is also faster to train and easier to interpret — the model coefficients directly tell us which features contribute most to the diagnosis. |

---

## Streamlit App Features

1. **Dataset upload option (CSV)** — Users can upload a test CSV file through the sidebar
2. **Model selection dropdown** — Dropdown menu to select any of the 5 trained models
3. **Display of evaluation metrics** — All 6 metrics shown as metric cards + comparison table
4. **Confusion matrix and classification report** — Heatmap confusion matrix + per-class metrics

---

## How to Run Locally

```bash
git clone https://github.com/yasukovi1753/ml-classification-app.git
cd ml-classification-app
pip install -r requirements.txt
cd model
python train_models.py
cd ..
streamlit run app.py
```
