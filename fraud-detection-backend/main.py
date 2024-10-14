import pandas as pd
from eda import exploratory_data_analysis
from feature_engineering import feature_engineering
from models import train_and_evaluate_models

def main():
    # Load data
    data = pd.read_csv('paysim.csv')
    
    # Exploratory Data Analysis
    exploratory_data_analysis(data)
    
    # Feature Engineering
    df = feature_engineering(data)
    
    # Train and Evaluate Models
    train_and_evaluate_models(df)

if __name__ == '__main__':
    main()
