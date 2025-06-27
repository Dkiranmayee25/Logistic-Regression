# Logistic-Regression
Logistic Regression: Breast Cancer Classification
This project demonstrates how to perform binary classification using Logistic Regression on the Breast Cancer Wisconsin Diagnostic dataset.

Tools Used:
Python
Pandas – Data manipulation
NumPy – Numerical computations
Scikit-learn – Machine learning model and evaluation
Matplotlib & Seaborn – Data visualization

Dataset
The dataset contains features derived from digitized images of fine needle aspirate (FNA) of breast masses. It helps in predicting whether a tumor is:
Malignant (M) → 1
Benign (B) → 0
Dataset columns include measurements like radius, texture, perimeter, area, smoothness, etc.
Columns id and Unnamed: 32 are dropped as they are not useful for prediction.

Project Workflow
Load and inspect the dataset
Clean and preprocess:
  Drop unnecessary columns
  Encode the target (M as 1, B as 0)
  Standardize features
  Split into training and testing sets
  Train Logistic Regression model

Evaluate:
Confusion matrix
Classification report (accuracy, precision, recall, F1-score)
ROC-AUC curve
Tune threshold to improve recall
Visualize the sigmoid function used in logistic regression

Files Included:
logistic_regression_breast_cancer.py – Main Python script
README.md – Project overview
Dataset - Breast Cancer sataset - (data.csv)
Output:
Confusion matrix heatmap
Classification report in the console
ROC curve
Custom threshold evaluation
Sigmoid curve plot
