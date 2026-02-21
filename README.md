❤️ Heart Disease Risk Prediction API
📌 Overview

This project implements an end-to-end Machine Learning pipeline to predict the risk of heart disease using clinical patient data.

The system:

Performs automated preprocessing using scikit-learn Pipelines

Applies Logistic Regression for classification

Evaluates performance using Cross-Validation, ROC-AUC, and Calibration curves

Optimizes classification threshold using Youden’s J statistic

Exposes predictions through a REST API built with Flask

📊 Dataset

The dataset contains clinical attributes such as:

Age

Sex

Chest pain type (cp)

Resting blood pressure

Cholesterol

Fasting blood sugar

ECG results

Maximum heart rate

Exercise-induced angina

ST depression (oldpeak)

Slope

Number of major vessels (ca)

Thalassemia type

Target variable:

0 → No heart disease

1 → Presence of heart disease

⚙️ Methodology
1️⃣ Data Preprocessing

All preprocessing is handled inside a scikit-learn Pipeline:

Median imputation for numerical features

Most-frequent imputation for categorical features

Standard scaling for numerical data

One-hot encoding for categorical variables

This ensures:

Reproducibility

No data leakage

Clean deployment integration

2️⃣ Model

Classifier used:

Logistic Regression (max_iter=1000)

Why Logistic Regression?

Interpretable

Stable on small medical datasets

Produces calibrated probability outputs

3️⃣ Model Evaluation
Cross-Validation (5-fold)

Mean ROC-AUC: 0.9182
Standard Deviation: 0.0218

This indicates strong generalization and low variance across folds.

Test Set Performance

Accuracy: 86.67%

ROC-AUC: 0.9529

Confusion Matrix:

[[29  4]
 [ 4 23]]

Only 4 false negatives — an important consideration in medical risk prediction.

4️⃣ Threshold Optimization

Instead of using the default 0.5 threshold, the classification threshold was optimized using Youden’s J statistic:

Optimal Threshold: 0.6593

This balances sensitivity and specificity for better clinical relevance.

5️⃣ Calibration Analysis

A calibration curve was generated to evaluate how well predicted probabilities reflect true outcome likelihood.

This ensures probability outputs are meaningful for decision-making.

🚀 API Deployment

The trained pipeline is serialized using pickle and served via Flask.

Start Server
python app.py

Server runs on:

http://127.0.0.1:5000
📥 Example API Request

POST /predict

{
  "age": 54,
  "sex": "Male",
  "cp": "asymptomatic",
  "trestbps": 140,
  "chol": 250,
  "fbs": false,
  "restecg": "normal",
  "thalch": 150,
  "exang": false,
  "oldpeak": 1.5,
  "slope": "flat",
  "ca": 0,
  "thal": "normal"
}
📤 Example API Response
{
  "prediction": 0,
  "risk_level": "Low",
  "probability_of_disease": 0.4101
}
📂 Project Structure
disease-risk-predictor/
│
├── heart.csv
├── disease_predictor.py
├── app.py
├── heart_model.pkl
├── roc_curve.png
├── calibration_curve.png
├── requirements.txt
└── README.md
🔮 Future Improvements

Experiment with ensemble models (XGBoost / LightGBM)

Add a frontend web interface

Deploy publicly (Render / Railway / Azure)

Add SHAP-based model explainability

📜 License

Educational / Research Use.