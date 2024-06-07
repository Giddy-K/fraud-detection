import pandas as pd

def feature_engineering(data):
    # Filter 'TRANSFER' and 'CASH_OUT' types
    data_by_type = data[data['type'].isin(['TRANSFER', 'CASH_OUT'])]
    
    # Subsample data
    df = data_by_type.sample(n=100000, random_state=1).reset_index(drop=True)
    
    # Drop name columns
    df = df.drop(['nameOrig', 'nameDest'], axis=1)
    
    # Binary-encoding of labelled data in 'type'
    df.loc[df.type == 'CASH_OUT', 'type'] = 1
    df.loc[df.type == 'TRANSFER', 'type'] = 0
    
    return df
