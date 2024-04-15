import React, { useState } from 'react';
import "./componentStyles.css";

export default function ImageUpload({ onUpload }) {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = () => {
        if (selectedFile) {
            onUpload(selectedFile);
            setSelectedFile(null); // Clear the selected file after upload
        }
    };

    return (
        <div className="uploadContainer">
            <div className="fileInputContainer">
                <label htmlFor="imageUpload" className="customFileInput">
                    <span>Choose an image</span>
                    <input type="file" id="imageUpload" accept="image/*" onChange={handleFileChange} />
                </label>
            </div>
            {selectedFile && (
                <div className="fileName">
                    Selected File: {selectedFile.name}
                </div>
            )}
            <button className="uploadButton" onClick={handleUpload}>Upload</button>
        </div>
    );
}
