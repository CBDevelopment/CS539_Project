"use client";

import React from 'react';
import ImageUpload from '../components/ImageUpload';

export default function Home() {
    const handleUpload = (file) => {
        console.log('File uploaded:', file);
    };

    return (
        <main className="centeredContainer">
            <h1 className="title">Where was this image taken?</h1>

            {/* Render the ImageUpload component */}
            <ImageUpload onUpload={handleUpload} />
        </main>
    );
}
