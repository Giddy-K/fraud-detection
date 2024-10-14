import seaborn as sns
import matplotlib.pyplot as plt
import os

def perform_eda(data):
    plt.figure(figsize=(10, 3))
    sns.histplot(data[data['isFraud'] == 0].step, label="Safe Transaction", kde=True)
    sns.histplot(data[data['isFraud'] == 1].step, label="Fraud Transaction", kde=True)
    plt.legend()

    # Save the plot as a PNG file
    output_folder = 'static'  # Create a 'static' folder to store the image
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_path = os.path.join(output_folder, 'eda_result.png')
    plt.savefig(image_path)  # Save as PNG

    return image_path
