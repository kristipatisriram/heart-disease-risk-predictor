Heart Disease Risk Predictor using Machine Learning

Project Overview

This project aims to predict the presence of heart disease in patients using machine learning techniques. The dataset used is based on the UCI Heart Disease dataset, which contains medical attributes such as age, cholesterol levels, chest pain type, and maximum heart rate achieved.

The objective of this project is to build a classification model that can accurately identify patients at risk of heart disease.

Machine Learning Pipeline

The following steps were performed:

Data cleaning and preprocessing

Handling missing values

Binary conversion of target variable

One-hot encoding of categorical features

Feature scaling using StandardScaler

Model training using:

Logistic Regression

Random Forest Classifier

K-Nearest Neighbors (KNN)

Model evaluation using:

Accuracy Score

Confusion Matrix

Precision, Recall, and F1-score

Feature importance analysis using Random Forest

Results
Model	Accuracy
Logistic Regression	84.7%
Random Forest	84.7%
KNN (k=5)	88.1%

The KNN classifier demonstrated the best performance with an accuracy of 88.13% and successfully minimized false negatives in disease detection.

Important Features Identified

Feature importance analysis revealed the following key predictors:

Number of major vessels (ca)

Maximum heart rate achieved (thalach)

ST depression induced by exercise (oldpeak)

Thalassemia test results (thal)

Age and cholesterol levels

Model Saving

The trained KNN model has been saved using Pickle for future deployment.

How to Run

1. Clone the repository
2. Install required dependencies:
pip install -r requirements.txt
3. Run the script:
python disease_predictor.py

Dataset:

UCI Heart Disease Dataset
https://archive.ics.uci.edu/ml/datasets/Heart+

Author
Kristipati Sri Ram