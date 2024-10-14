import React, { useState } from 'react';
import axios from 'axios';

const Upload = () => {
    const [file, setFile] = useState(null);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('/api/upload', formData);
            console.log(response.data);
        } catch (error) {
            console.error('Error uploading file', error);
        }
    };

    return (
        <div>
            <h2>Upload Dataset</h2>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload</button>
        </div>
    );
};

export default Upload;
