# 💳 Loan Payback Prediction

A machine learning pipeline that predicts whether a borrower will pay back their loan, built for the [Kaggle Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11) competition.

---

## 📌 Overview

This project tackles a binary classification problem in consumer lending: given a borrower's demographic profile, financial standing, and loan details, predict whether they will pay back their loan. The pipeline covers data loading, feature engineering, label encoding, model training with a tuned Random Forest, and submission generation.

---

## 🗂️ Dataset

The dataset is sourced from the Kaggle Playground Series S5E11 competition and includes the following features:

| Category | Features |
|---|---|
| Demographics | `gender`, `marital_status`, `education_level` |
| Employment | `employment_status`, `annual_income` |
| Loan Details | `loan_amount`, `loan_purpose`, `grade_subgrade` |
| Engineered | `loan_to_income_ratio` |
| Target | `loan_paid_back` (binary: paid back / not paid back) |

---

## 🔧 Tech Stack

- **Python 3**
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — preprocessing, modelling, evaluation
- `matplotlib`, `seaborn`, `plotnine` — visualisation
- Kaggle environment

---

## 🚀 Pipeline

### 1. Data Loading
Train and test CSVs are loaded separately from the Kaggle input directory.

### 2. Exploratory Data Analysis
Shape, data types, descriptive statistics, and class distribution of the target variable are reviewed for both datasets.

### 3. Feature Engineering
A `loan_to_income_ratio` is derived by dividing `loan_amount` by `annual_income + 1` (the +1 avoids division by zero). This single ratio captures the borrower's debt burden relative to their earnings and is applied consistently to both train and test sets.

### 4. Preprocessing
- The `id` column is dropped from the training set
- `LabelEncoder` is applied to all categorical features: `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, and `grade_subgrade`

### 5. Train / Validation Split
Stratified 80/20 split with `random_state=42` to preserve the class ratio across both sets.

### 6. Model Training
A Random Forest Classifier is trained with the following configuration:

```python
RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
```

`class_weight='balanced'` handles any imbalance between borrowers who paid back and those who did not.

### 7. Feature Importance Visualisation
A horizontal bar chart plots Mean Gini Decrease per feature. Features exceeding an importance threshold of 0.08 are highlighted in a distinct colour.

### 8. Submission
Predictions on the test set are saved to `submission.csv` with `id` and `loan_paid_back` columns.

---

## 📊 Evaluation

Model performance is reported using:

- Training and validation accuracy
- Out-of-bag (OOB) score
- Full classification report: precision, recall, and F1-score for both `Paid back` and `Not Paid back` classes

---

## 📁 Repository Structure

```
├── loan-payback-prediction.ipynb   # Main notebook
├── submission.csv                  # Kaggle submission output
└── README.md
```

---

## ▶️ How to Run

### On Kaggle
1. Add the [Playground Series S5E11 dataset](https://www.kaggle.com/competitions/playground-series-s5e11/data) to your notebook
2. Open and run all cells
3. `submission.csv` will be saved to the working directory

### Locally
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotnine
jupyter notebook loan-payback-prediction.ipynb
```

Update the file paths in the data loading cells from `/kaggle/input/...` to your local paths.

---

## 🔭 Potential Improvements

- Add more engineered features such as debt-to-credit ratio or income brackets
- Explore XGBoost or LightGBM for potentially stronger performance
- Apply hyperparameter tuning via `GridSearchCV` (already imported)
- Use SMOTE or other resampling strategies if class imbalance is significant
- Add ROC-AUC curve visualisation (imports for `roc_curve` and `auc` are already in place)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
