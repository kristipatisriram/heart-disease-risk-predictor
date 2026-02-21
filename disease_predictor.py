# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Drop useless columns
# df = df.drop(['id','dataset'], axis=1)

# # Convert num to binary
# df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# # Rename target
# df.rename(columns={'num':'target'}, inplace=True)

# # Detect categorical columns automatically
# categorical_cols = df.select_dtypes(include=['object']).columns
# print("Categorical Columns:", categorical_cols)

# # Encode ALL categorical columns
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# print(df['target'].value_counts())

# X = df.drop('target', axis=1)
# y = df['target']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# lr_pred = lr.predict(X_test)

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)

# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)

# print("Logistic Regression:", accuracy_score(y_test, lr_pred))
# print("Random Forest:", accuracy_score(y_test, rf_pred))
# print("KNN:", accuracy_score(y_test, knn_pred))



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV


# # ==========================
# # 1. LOAD DATASET
# # ==========================
# df = pd.read_csv("heart.csv", na_values='?')

# print("First five rows:")
# print(df.head())

# print("\nMissing values:\n", df.isnull().sum())


# # ==========================
# # 2. HANDLE MISSING VALUES
# # ==========================
# df = df.dropna()

# print("\nDataset Shape:", df.shape)


# # ==========================
# # 3. DROP USELESS COLUMNS
# # ==========================
# df = df.drop(['id','dataset'], axis=1)


# # ==========================
# # 4. CONVERT TARGET TO BINARY
# # ==========================
# df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# df.rename(columns={'num':'target'}, inplace=True)

# print("\nTarget Distribution:")
# print(df['target'].value_counts())


# # ==========================
# # 5. ENCODE CATEGORICAL DATA
# # ==========================
# categorical_cols = df.select_dtypes(include=['object']).columns
# print("\nCategorical Columns:", categorical_cols)

# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# # ==========================
# # 6. SPLIT FEATURES & TARGET
# # ==========================
# X = df.drop('target', axis=1)
# y = df['target']


# # ==========================
# # 7. TRAIN TEST SPLIT
# # ==========================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )


# # ==========================
# # 8. FEATURE SCALING
# # ==========================
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# #defining k range
# param_grid = {
#     'n_neighbors': list(range(1,21))
# }

# #grid search setup
# grid = GridSearchCV(
#     KNeighborsClassifier(),
#     param_grid,
#     cv=5,
#     scoring='accuracy'
# )


# # ==========================
# # 9. TRAIN MODELS
# # ==========================

# # Logistic Regression
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# lr_pred = lr.predict(X_test)

# # Random Forest
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)

# # KNN
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)
# grid.fit(X_train, y_train)
# print("Best K Value:", grid.best_params_)
# best_knn = grid.best_estimator_

# best_knn_pred = best_knn.predict(X_test)


# # ==========================
# # 10. COMPARE ACCURACY
# # ==========================
# print("\nLogistic Regression:", accuracy_score(y_test, lr_pred))
# print("Random Forest:", accuracy_score(y_test, rf_pred))
# print("KNN:", accuracy_score(y_test, knn_pred))
# print("Tuned KNN Accuracy:", accuracy_score(y_test, best_knn_pred))

# #confusion matrix
# from sklearn.metrics import confusion_matrix, classification_report

# print("\nConfusion Matrix (KNN):")
# print(confusion_matrix(y_test, knn_pred))

# print("\nClassification Report (KNN):")
# print(classification_report(y_test, knn_pred))

# print(confusion_matrix(y_test, best_knn_pred))
# print(classification_report(y_test, best_knn_pred))


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # ==========================
# # 1. LOAD DATASET
# # ==========================
# df = pd.read_csv("heart.csv", na_values='?')

# print("First five rows:")
# print(df.head())

# print("\nMissing values:\n", df.isnull().sum())


# # ==========================
# # 2. HANDLE MISSING VALUES
# # ==========================
# df = df.dropna()
# print("\nDataset Shape:", df.shape)


# # ==========================
# # 3. DROP USELESS COLUMNS
# # ==========================
# df = df.drop(['id','dataset'], axis=1)


# # ==========================
# # 4. CONVERT TARGET TO BINARY
# # ==========================
# df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
# df.rename(columns={'num':'target'}, inplace=True)

# print("\nTarget Distribution:")
# print(df['target'].value_counts())


# # ==========================
# # 5. ENCODE CATEGORICAL DATA
# # ==========================
# categorical_cols = df.select_dtypes(include=['object']).columns
# print("\nCategorical Columns:", categorical_cols)

# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# # ==========================
# # 6. SPLIT FEATURES & TARGET
# # ==========================
# X = df.drop('target', axis=1)
# y = df['target']


# # ==========================
# # 7. TRAIN TEST SPLIT
# # ==========================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )


# # ==========================
# # 8. FEATURE SCALING
# # ==========================
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # ==========================
# # 9. TRAIN MODELS
# # ==========================

# # Logistic Regression
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)
# lr_pred = lr.predict(X_test)


# # Random Forest (Tuned)
# rf = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=6,
#     random_state=42
# )
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)


# # KNN (Best practical value)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)


# # ==========================
# # 10. COMPARE ACCURACY
# # ==========================
# print("\nLogistic Regression:", accuracy_score(y_test, lr_pred))
# print("Random Forest:", accuracy_score(y_test, rf_pred))
# print("KNN:", accuracy_score(y_test, knn_pred))


# # ==========================
# # 11. CONFUSION MATRIX
# # ==========================
# print("\nConfusion Matrix (KNN):")
# print(confusion_matrix(y_test, knn_pred))

# print("\nClassification Report (KNN):")
# print(classification_report(y_test, knn_pred))


# # ==========================
# # 12. FEATURE IMPORTANCE
# # ==========================
# feature_importances = rf.feature_importances_
# feature_names = X.columns

# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': feature_importances
# }).sort_values(by='Importance', ascending=False)

# print("\nTop 10 Important Features:\n")
# print(importance_df.head(10))


# # ==========================
# # 13. SAVE MODEL
# # ==========================
# with open("knn_model.pkl", "wb") as f:
#     pickle.dump(knn, f)

# print("\nModel saved successfully!")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import roc_curve, auc
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

# # ==========================
# # 1. LOAD DATASET
# # ==========================
# df = pd.read_csv("heart.csv", na_values='?')

# print("First five rows:")
# print(df.head().to_string())

# print("\nMissing values:\n", df.isnull().sum())


# # ==========================
# # 2. HANDLE MISSING VALUES
# # ==========================

# # Separate numeric and categorical columns
# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
# categorical_cols = df.select_dtypes(include=['object']).columns

# # Median for numeric
# num_imputer = SimpleImputer(strategy='median')
# df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# # Most frequent for categorical
# cat_imputer = SimpleImputer(strategy='most_frequent')
# df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# print("\nDataset Shape:", df.shape)


# # ==========================
# # 3. DROP USELESS COLUMNS
# # ==========================
# df = df.drop(['id','dataset'], axis=1)


# # ==========================
# # 4. CONVERT TARGET TO BINARY
# # ==========================
# df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
# df.rename(columns={'num':'target'}, inplace=True)

# print("\nTarget Distribution:")
# print(df['target'].value_counts())


# # ==========================
# # 5. ENCODE CATEGORICAL DATA
# # ==========================
# categorical_cols = df.select_dtypes(include=['object']).columns
# print("\nCategorical Columns:", categorical_cols)

# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# # ==========================
# # 6. SPLIT FEATURES & TARGET
# # ==========================
# X = df.drop('target', axis=1)
# y = df['target']


# # ==========================
# # 7. TRAIN TEST SPLIT
# # ==========================
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)


# # ==========================
# # 8. FEATURE SCALING
# # ==========================
# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # ==========================
# # 9. TRAIN MODELS
# # ==========================

# # Logistic Regression
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)
# lr_pred = lr.predict(X_test)


# # Random Forest (Tuned)
# rf = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=6,
#     random_state=42
# )
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)


# # KNN (Best practical value)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# knn_pred = knn.predict(X_test)


# # ==========================
# # 10. COMPARE ACCURACY
# # ==========================
# print("\nLogistic Regression:", accuracy_score(y_test, lr_pred))
# print("Random Forest:", accuracy_score(y_test, rf_pred))
# print("KNN:", accuracy_score(y_test, knn_pred))


# # ==========================
# # 11. CONFUSION MATRIX
# # ==========================
# print("\nConfusion Matrix (KNN):")
# print(confusion_matrix(y_test, knn_pred))

# print("\nClassification Report (KNN):")
# print(classification_report(y_test, knn_pred))


# # ==========================
# # 12. FEATURE IMPORTANCE
# # ==========================
# feature_importances = rf.feature_importances_
# feature_names = X.columns

# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': feature_importances
# }).sort_values(by='Importance', ascending=False)

# print("\nTop 10 Important Features:\n")
# print(importance_df.head(10))


# # ==========================
# # 13. SAVE MODEL
# # ==========================
# with open("knn_model.pkl", "wb") as f:
#     pickle.dump({
#         "model":lr,
#         "scaler":scaler
#     },f)

# print("\nModel saved successfully!")

# # ==========================
# # 14. ROC CURVE
# # ==========================
# knn_probs = knn.predict_proba(X_test)[:, 1]

# fpr, tpr, thresholds = roc_curve(y_test, knn_probs)
# roc_auc = auc(fpr, tpr)

# print("\nROC AUC Score:", roc_auc)

# plt.figure()
# plt.plot(fpr, tpr, label=f'KNN (AUC = {roc_auc:.2f})')
# plt.plot([0,1], [0,1], linestyle='--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.savefig("roc_curve.png")
# plt.close()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.calibration import calibration_curve


# # ==========================
# # 1. LOAD DATASET
# # ==========================
# df = pd.read_csv("heart.csv", na_values='?')

# print("First five rows:")
# print(df.head().to_string())

# print("\nMissing values:\n", df.isnull().sum())


# # ==========================
# # 2. DROP USELESS COLUMNS
# # ==========================
# df = df.drop(['id', 'dataset'], axis=1)


# # ==========================
# # 3. CONVERT TARGET TO BINARY
# # ==========================
# df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
# df.rename(columns={'num': 'target'}, inplace=True)

# print("\nTarget Distribution:")
# print(df['target'].value_counts())


# # ==========================
# # 4. SEPARATE FEATURES & TARGET
# # ==========================
# X = df.drop('target', axis=1)
# y = df['target']


# # Identify column types
# numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_cols = X.select_dtypes(include=['object', 'bool']).columns


# # ==========================
# # 5. BUILD PREPROCESSING PIPELINE
# # ==========================

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(drop='first'))
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ]
# )


# # ==========================
# # 6. BUILD FULL MODEL PIPELINE
# # ==========================

# model_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', LogisticRegression(max_iter=1000))
# ])
# scores = cross_val_score(...)

# # ==========================
# # 7. TRAIN TEST SPLIT
# # ==========================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )


# # ==========================
# # 8. TRAIN MODEL
# # ==========================
# model_pipeline.fit(X_train, y_train)

# y_pred = model_pipeline.predict(X_test)


# # ==========================
# # 9. EVALUATION
# # ==========================
# print("\nAccuracy:", accuracy_score(y_test, y_pred))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='roc_auc')

# print("\nCross-Validation AUC Scores:", scores)
# print("Mean CV AUC:", round(scores.mean(), 4))
# print("Std Dev CV AUC:", round(scores.std(), 4))

# # ==========================
# # 10. ROC CURVE
# # ==========================
# y_probs = model_pipeline.predict_proba(X_test)[:, 1]

# fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# # Optimal threshold using Youden's J statistic
# j_scores = tpr - fpr
# optimal_idx = np.argmax(j_scores)
# optimal_threshold = thresholds[optimal_idx]

# print("\nOptimal Threshold:", round(optimal_threshold, 4))
# roc_auc = auc(fpr, tpr)

# print("\nROC AUC Score:", roc_auc)

# plt.figure()
# plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.savefig("roc_curve.png")
# plt.close()



# # ==========================
# # 11. CALIBRATION CURVE
# # ==========================
# prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

# plt.figure()
# plt.plot(prob_pred, prob_true, marker='o')
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel("Mean Predicted Probability")
# plt.ylabel("True Probability")
# plt.title("Calibration Curve")
# plt.savefig("calibration_curve.png")
# plt.close()

# print("Calibration curve saved.")

# # ==========================
# # 11. SAVE FULL PIPELINE
# # ==========================
# with open("heart_model.pkl", "wb") as f:
#     pickle.dump(model_pipeline, f)

# print("\nFull pipeline saved successfully!")




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