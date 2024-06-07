# Machine Learning Model for Fraud Detection in Mobile Transactions
# Project Overview
The goal of this project is to develop and optimize machine learning models to detect fraudulent mobile transactions using the Paysim dataset from Kaggle. The models tested include Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, XGBoost, and Support Vector Machine (SVM). The top-performing models, XGBoost and Random Forest, were further optimized through hyperparameter tuning and techniques to handle imbalanced data.

# Data
The dataset used is the Paysim dataset from Kaggle, which simulates mobile money transactions based on real transaction logs from a mobile financial service in an African country. Key columns include transaction type, amount, customer identifiers, balance information, and fraud labels.

Project Steps
Loading Data and EDA
Feature Engineering
Machine Learning
3.1. Baseline Models
3.2. Grid Search for Best Hyper-parameters
3.3. Dealing with Unbalanced Data
3.3.1. Balancing Data via Oversampling with SMOTE
3.3.2. Subsampling Data from the Original Dataset
3.3.3. Performing SMOTE on the New Data
Machine Learning Pipeline
Feature Importance
Conclusion
Future Works
1. Loading Data and EDA
Code
pyt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data 
data = pd.read_csv('paysim.csv')

# Check for null values and duplicates
print(data.isna().sum().sum())  # 0
print(data.duplicated(keep='first').any())  # False

# Filter data by labels
safe = data[data['isFraud'] == 0]
fraud = data[data['isFraud'] == 1]

# Distribution of transactions over time
plt.figure(figsize=(10, 3))
sns.histplot(safe.step, label="Safe Transaction", kde=True)
sns.histplot(fraud.step, label='Fraud Transaction', kde=True)
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Transactions over the Time')
plt.legend()
plt.show()

# Hourly transaction amounts
smalldata = data.sample(n=100000, random_state=1).reset_index(drop=True)
plt.figure(figsize=(18, 6))
plt.ylim(0, 10000000)
plt.title('Hourly Transaction Amounts')
sns.scatterplot(x="step", y="amount", hue="isFraud", data=smalldata)
plt.show()

# Hourly fraud transaction amounts
plt.figure(figsize=(18, 6))
plt.ylim(0, 10000000)
plt.title('Hourly Fraud Transaction Amounts')
sns.scatterplot(x="step", y="amount", color='orange', data=fraud)
plt.show()
2. Feature Engineering
Code
pyt

# Filter only 'TRANSFER' and 'CASH_OUT' types
data_by_type = data[data['type'].isin(['TRANSFER', 'CASH_OUT'])]

# Subsample data to 100,000 instances
df = data_by_type.sample(n=100000, random_state=1).reset_index(drop=True)

# Drop name columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Binary-encoding of labelled data in 'type'
df.loc[df.type == 'CASH_OUT', 'type'] = 1
df.loc[df.type == 'TRANSFER', 'type'] = 0
3. Machine Learning
3.1. Baseline Models
Code
pyt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score, classification_report

# Slice target and features
features = df.drop('isFraud', axis=1)
target = df.isFraud

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# List of classifiers
classifiers = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, XGBClassifier, svm.SVC]

# Function to train and evaluate classifiers
def ml_func(algorithm):
    model = algorithm()
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_accuracy = roc_auc_score(y_train, train_preds)
    test_accuracy = roc_auc_score(y_test, test_preds)
    print(f"{algorithm.__name__}: Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Train and evaluate each classifier
for classifier in classifiers:
    ml_func(classifier)
3.2. Grid Search for Best Hyper-parameters
Code
pyt

from sklearn.model_selection import GridSearchCV

# Grid search function
def grid_src(classifier, param_grid):
    grid_search = GridSearchCV(classifier, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"{classifier.__class__.__name__} Best Parameters: {grid_search.best_params_}")
    return grid_search.best_params_

# Grid search for RandomForestClassifier
param_grid_rf = {'n_estimators': [10, 80, 100], 'criterion': ['gini', 'entropy'], 'max_depth': [10], 'min_samples_split': [2, 3, 4]}
rf_params = grid_src(RandomForestClassifier(), param_grid_rf)

# Grid search for XGBClassifier
param_grid_xg = {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 10], 'colsample_bytree': [0.7, 1], 'gamma': [0.0, 0.1, 0.2]}
xgb_params = grid_src(XGBClassifier(), param_grid_xg)
3.3. Dealing with Unbalanced Data
3.3.1. Balancing Data via Oversampling with SMOTE
Code
pyt

from imblearn.over_sampling import SMOTE

# Resample data using SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

# Train-test split on resampled data
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_resampled, y_resampled, random_state=0)

# Running models with balanced data
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, min_samples_split=3)
xgb_model = XGBClassifier(colsample_bytree=1, n_estimators=100, gamma=0.1, learning_rate=0.1, max_depth=5)

run_model(rf_model, X_train_smote, y_train_smote, X_test_smote, y_test_smote)
run_model(xgb_model, X_train_smote, y_train_smote, X_test_smote, y_test_smote)
3.3.2. Subsampling Data from the Original Dataset
Code
pyt

# Filter 'TRANSFER' and 'CASH_OUT' types
data2 = data[data['type'].isin(['TRANSFER', 'CASH_OUT'])]
safe_2 = data2[data2['isFraud'] == 0]
fraud_2 = data2[data2['isFraud'] == 1]

# Sample 50,000 safe transactions
safe_sample = safe_2.sample(n=50000, random_state=1).reset_index(drop=True)

# Combine sampled safe transactions and all fraud transactions
df3 = pd.concat([safe_sample, fraud_2]).reset_index(drop=True)

# Drop name columns
df3 = df3.drop(['nameOrig', 'nameDest'], axis=1)

# Binary-encoding of labelled data in 'type'
df3.loc[df3.type == 'CASH_OUT', 'type'] = 1
df3.loc[df3.type == 'TRANSFER', 'type'] = 0

# Slice target and features
features2 = df3.drop('isFraud', axis=1)
target2 = df3.isFraud

# Split data into train and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, target2, test_size=0.2)

# Running models with subsampled organic data
run_model(rf_model, X_train2, y_train2, X_test2, y_test2)
run_model(xgb_model, X_train2, y_train2, X_test2, y_test2)
4. Machine Learning Pipeline
The final pipeline includes data preprocessing, feature engineering, model training, and evaluation. The best-performing models, XGBoost and RandomForest, were optimized and trained on both SMOTE-balanced and subsampled data to handle the class imbalance issue effectively.

5. Feature Importance
Code
pyt

# Feature importance for RandomForest
import numpy as np
import matplotlib.pyplot as plt

importances_rf = rf_model.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
plt.figure()
plt.title("Feature importances (RandomForest)")
plt.bar(range(X_train.shape[1]), importances_rf[indices_rf], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices_rf], rotation=90)
plt.show()

# Feature importance for XGBoost
importances_xgb = xgb_model.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1]
plt.figure()
plt.title("Feature importances (XGBoost)")
plt.bar(range(X_train.shape[1]), importances_xgb[indices_xgb], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices_xgb], rotation=90)
plt.show()
6. Conclusion
The XGBoost and RandomForest models demonstrated robust performance in detecting fraudulent transactions, especially after addressing class imbalance with techniques like SMOTE and subsampling. Future work could focus on further model tuning, testing additional algorithms, and implementing the model in a real-time transaction monitoring system.

7. Future Works
Ensemble Methods: Combining multiple models to further improve prediction accuracy.
Real-time Deployment: Implementing the model in a real-time fraud detection system.
Feature Engineering: Creating more sophisticated features to better capture transaction patterns.
Model Interpretability: Using methods like SHAP (SHapley Additive exPlanations) for better model interpretability.
Setup and Installation
Prerequisites
Python 3.x
Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, imbalanced-learn
Installation
Clone the repository:


git clone https://github.com/Giddy-K/fraud-detection.git
cd fraud-detection
Create a virtual environment:


python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
Install the required packages:


pip install -r requirements.txt
Running the Project
Place the paysim.csv dataset in the project directory.
Run the main script:

python main.py
This will execute the entire pipeline from data loading, EDA, feature engineering, model training, and evaluation, as well as visualizing feature importance.

This README provides a comprehensive overview of the project, setup instructions, and detailed steps for data loading, feature engineering, model training, and evaluation. Follow the instructions to replicate the results and further explore the dataset and models.