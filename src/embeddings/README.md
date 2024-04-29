# embeddings
## Author: Cutter Beck

This directory is used to create embeddings to use for regression output of a location based on an input image

MAKE SURE TO LOAD IN THE `image_embeddings.npy` FILE FROM THE DRIVE

### Files
- `autoencoder.py`
    - Defines an Autoencoder class which expands GPS coordinates of shape 1x2 to 1x20 and reduces down to 1 number as an embedding
- `coordEmbedding.py`
    - Trains and saves the autoencoder
- `coordPrediction.py`
    - Linear Regression between image embeddings and GPS embeddings. Displays the actual and predicted results of the regression after decoding the GPS coordinates through the autoencoder
- `imageEmbedding.py`
    - Implements the pretrained EfficientNet-b5 model from Google to create embeddings of images
        - Can be manipulated to run on either Turing cluster or Google Colab
            - Cutter ran on Colab in ~40 min on 1 A100
- `visualizeVectors.py`
    - Visualizes large dimensionality vectors in 2D space using PCA