from flask import Flask, request, jsonify, send_file
import pandas as pd
from eda import perform_eda

app = Flask(__name__)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        data = pd.read_csv(file)
        image_path = perform_eda(data)
        return jsonify({"message": "EDA performed successfully", "image_url": f'http://127.0.0.1:5000/{image_path}'})

if __name__ == '__main__':
    app.run(debug=True)
