from flask import Flask, request, jsonify
from feature_engineering import feature_engineering
from eda import exploratory_data_analysis
from models import train_and_evaluate_models
import pandas as pd
import os

app = Flask(__name__)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file)
    
    # Perform EDA
    exploratory_data_analysis(data)
    
    # Feature Engineering
    df = feature_engineering(data)
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(df)

    return jsonify(model_results)

@app.route('/api/predict', methods=['POST'])
def predict():
    # Extract features from the request
    features = request.json
    # Add your prediction logic here (assuming a trained model is available)
    # Example:
    # prediction = model.predict(features)
    return jsonify({'prediction': 'your prediction here'})

if __name__ == '__main__':
    app.run(debug=True)