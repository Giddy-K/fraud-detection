import React from 'react';

const Results = ({ results }) => {
    return (
        <div>
            <h2>Results</h2>
            {/* Display EDA results and model evaluation metrics */}
            <p>{JSON.stringify(results)}</p>
        </div>
    );
};

export default Results;
