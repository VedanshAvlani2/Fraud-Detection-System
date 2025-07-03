# Fraud Detection System with Time-Based Retraining

## 🧠 Overview
This project implements a time-aware fraud detection model using the popular Credit Card Fraud dataset. It simulates real-world deployment by retraining on rolling time windows and validating performance on future, unseen transactions.

## 🎯 Project Objective
- Detect credit card fraud with high precision and recall.
- Simulate production-like scheduled model retraining.
- Tune decision thresholds to optimize performance.

## 📂 Dataset & Inputs
- **Source**: `creditcard.csv` (Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud )
- **Features**: Time, V1–V28 (anonymized), Amount
- **Target**: `Class` (1 = Fraud, 0 = Normal)

## ⚙️ Technologies Used
- Python (pandas, numpy)
- Scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## 🏃 How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### 2. Run Script
```bash
python fraud_detection.py
```

## 🔁 Workflow

1. **Preprocessing**:
   - Scale data, handle missing values
2. **Retraining Simulation**:
   - 30 rolling time windows of 15,000 samples
   - Retrain using Random Forest on SMOTE-balanced data
3. **Future-Time Simulation**:
   - Train on 80% time-based data
   - Test on 20% future fraud cases
4. **Threshold Tuning**:
   - Compare metrics at different decision thresholds
5. **Visualization**:
   - Plot AUC over retraining steps

## 📈 Results Summary
- AUC: ~0.93 (Future-Time)
- Precision: Up to 0.96
- Recall: Tuned up to 0.77
- Best F1-Score: 0.82

## 🔍 Feature Importance (via Random Forest)
Feature importance is derived from model and can be used to reduce dimensions in future iterations.

## 🧠 Key Takeaways
- Rolling time-window training improves temporal generalization.
- Lower thresholds help detect more fraud but may reduce precision.
- AUC remains stable over time, validating retraining effectiveness.

## 🚀 Future Enhancements
- Add LightGBM/XGBoost comparisons
- Use LIME/SHAP for explainability
- Deploy with Streamlit or FastAPI

---
