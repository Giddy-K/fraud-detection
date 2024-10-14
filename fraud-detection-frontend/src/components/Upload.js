import React, { useState } from 'react';

const Upload = () => {
    const [imageUrl, setImageUrl] = useState(null);
    const [error, setError] = useState(null);

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://127.0.0.1:5000/api/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (data.image_url) {
                setImageUrl(data.image_url); // Set the image URL in state
            } else {
                setError(data.error);
            }
        } catch (err) {
            console.error(err);
            setError('An error occurred while uploading the file.');
        }
    };

    return (
        <div>
            <h2>Upload CSV File</h2>
            <input type="file" onChange={handleFileUpload} />
            
            {error && <p style={{ color: 'red' }}>{error}</p>}
            
            {imageUrl && (
                <div>
                    <h3>EDA Results:</h3>
                    <img src={imageUrl} alt="EDA Result" />
                </div>
            )}
        </div>
    );
};

export default Upload;
