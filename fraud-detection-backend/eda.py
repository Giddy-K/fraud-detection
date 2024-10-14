import seaborn as sns
import matplotlib.pyplot as plt

def exploratory_data_analysis(data):
    # Plot distribution of transactions over time
    plt.figure(figsize=(10, 3))
    sns.distplot(data[data['isFraud'] == 0].step, label="Safe Transaction")
    sns.distplot(data[data['isFraud'] == 1].step, label='Fraud Transaction')
    plt.xlabel('Hour')
    plt.ylabel('Number of Transactions')
    plt.title('Distribution of Transactions over the Time')
    plt.legend()
    plt.show()
    
    # Other plots and analyses...

    # Note: Add more EDA plots and analysis as needed
