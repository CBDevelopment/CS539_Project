# embeddings
## Author: Cutter Beck

This directory is used to create embeddings to use for regression output of a location based on an input image

### Files
- `coordEmbedding.py`
    - Defines an autoencoder which creates encoded representations of the the existing GPS coordinates with an output dimensionality of 1x1000 to match the dimensionality of the image embeddings
- `coordPrediction.py`
    - Attempt at doing a regression on the embeddings of both GPS coordinates and images to output predicted GPS coordinates
- `imageEmbedding.py`
    - Implements the pretrained EfficientNet-b5 model from Google to create embeddings of images
        - Takes a couple hours to run
            - Going to try running on Turing cluster/Colab A100s
- `visualizeVectors.py`
    - Visualizes large dimensionality vectors in 2D space using PCA