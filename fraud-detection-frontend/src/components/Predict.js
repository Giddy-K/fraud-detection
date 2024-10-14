import React, { useState } from 'react';
import axios from 'axios';

const Predict = () => {
    const [transaction, setTransaction] = useState({});
    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        setTransaction({
            ...transaction,
            [e.target.name]: e.target.value,
        });
    };

    const handlePredict = async () => {
        try {
            const response = await axios.post('/api/predict', transaction);
            setPrediction(response.data);
        } catch (error) {
            console.error('Error getting prediction', error);
        }
    };

    return (
        <div>
            <h2>Predict Fraudulent Transaction</h2>
            {/* Input fields for transaction data */}
            <input name="amount" onChange={handleChange} placeholder="Amount" />
            {/* Add other fields as necessary */}
            <button onClick={handlePredict}>Predict</button>
            {prediction && <p>Prediction: {prediction.isFraud ? 'Fraud' : 'Not Fraud'}</p>}
        </div>
    );
};

export default Predict;
