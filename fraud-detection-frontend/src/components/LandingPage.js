import React from 'react';
import { Link } from 'react-router-dom';

const LandingPage = () => {
    return (
        <div>
            <h1>Fraud Detection System</h1>
            <p>Welcome to the Fraud Detection System. Please upload your dataset to get started.</p>
            <Link to="/upload">
                <button>Upload Dataset</button>
            </Link>
        </div>
    );
};

export default LandingPage;
