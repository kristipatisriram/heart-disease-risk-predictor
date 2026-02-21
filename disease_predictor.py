import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


# ==========================
# 1. LOAD DATASET
# ==========================
df = pd.read_csv("heart.csv", na_values='?')

print("First five rows:")
print(df.head().to_string())

print("\nMissing values:\n", df.isnull().sum())


# ==========================
# 2. CLEAN DATA
# ==========================
df = df.drop(['id', 'dataset'], axis=1)

df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.rename(columns={'num': 'target'}, inplace=True)

print("\nTarget Distribution:")
print(df['target'].value_counts())


# ==========================
# 3. SEPARATE FEATURES & TARGET
# ==========================
X = df.drop('target', axis=1)
y = df['target']

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns


# ==========================
# 4. BUILD PREPROCESSING PIPELINE
# ==========================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# ==========================
# 5. BUILD MODEL PIPELINE
# ==========================
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# ==========================
# 6. CROSS-VALIDATION (Before Final Split)
# ==========================
cv_scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv=5,
    scoring='roc_auc'
)

print("\nCross-Validation AUC Scores:", cv_scores)
print("Mean CV AUC:", round(cv_scores.mean(), 4))
print("Std Dev CV AUC:", round(cv_scores.std(), 4))


# ==========================
# 7. TRAIN-TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# 8. TRAIN FINAL MODEL
# ==========================
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)


# ==========================
# 9. EVALUATION
# ==========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================
# 10. ROC CURVE + THRESHOLD
# ==========================
y_probs = model_pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Optimal threshold (Youden’s J)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print("\nOptimal Threshold:", round(optimal_threshold, 4))
print("Test ROC AUC:", round(roc_auc, 4))

plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()


# ==========================
# 11. CALIBRATION CURVE
# ==========================
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.savefig("calibration_curve.png")
plt.close()

print("Calibration curve saved.")


# ==========================
# 12. SAVE FULL PIPELINE
# ==========================
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("\nFull pipeline saved successfully!")