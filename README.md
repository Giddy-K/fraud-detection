# Machine Learning Model for Fraud Detection in Mobile Transactions

## Project Overview
The goal of this project is to develop and optimize machine learning models to detect fraudulent mobile transactions using the Paysim dataset from Kaggle. The models tested include Logistic Regression, K-Nearest Neighbors (KNN), Random Forest, XGBoost, and Support Vector Machine (SVM). The top-performing models, XGBoost and Random Forest, were further optimized through hyperparameter tuning and techniques to handle imbalanced data.

## Data
The dataset used is the Paysim dataset from Kaggle, which simulates mobile money transactions based on real transaction logs from a mobile financial service in an African country. Key columns include transaction type, amount, customer identifiers, balance information, and fraud labels.

## Project Steps
1. **Loading Data and EDA**
    - Code snippet for data loading and initial exploratory data analysis (EDA).
  
2. **Feature Engineering**
    - Steps for filtering and processing data to prepare it for modeling.

3. **Machine Learning**
    - 3.1. Baseline Models
        - Code to establish baseline models.
    - 3.2. Grid Search for Best Hyper-parameters
        - Hyperparameter tuning using GridSearchCV.
    - 3.3. Dealing with Unbalanced Data
        - 3.3.1. Balancing Data via Oversampling with SMOTE
        - 3.3.2. Subsampling Data from the Original Dataset
        - 3.3.3. Performing SMOTE on the New Data. 

4. **Machine Learning Pipeline**
    - Overview of the final pipeline including data preprocessing, feature engineering, model training, and evaluation.

5. **Feature Importance**
    - Visualization of feature importance for RandomForest and XGBoost models.

6. **Conclusion**
    - Summary of findings and model performance.

7. **Future Works**
    - Ideas for future enhancements, including ensemble methods, real-time deployment, feature engineering, and model interpretability.

## Setup and Installation

### Prerequisites
- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, imbalanced-learn

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Giddy-K/fraud-detection.git
    cd fraud-detection
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
- Place the `paysim.csv` dataset in the project directory.
- Run the main script:
    ```bash
    python main.py
    ```

This will execute the entire pipeline from data loading, EDA, feature engineering, model training, and evaluation, as well as visualizing feature importance.

## Acknowledgments
- [Kaggle](https://www.kaggle.com/) for the Paysim dataset.
