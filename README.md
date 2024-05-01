# CS539_Project
## Authors
- Michael Alicea
- Anushka Bangal
- Cutter Beck
- Blake Bruell
- Edward Smith

# Image Geolocation
- [Website](https://sites.google.com/view/image-geolocation)
- There are READMEs in most directories of this project that provide descriptions of the included files

# Running Models
## Requirements
- Python 3.8
- PyTorch
- Numpy
- Matplotlib
- Scikit-learn
- transformers (HuggingFace)
- pickle
- PIL

## Model 1: City Image Classifier
- Go to `src/cityMultiClassifier`
- Open `customDataset.py` and change the `DATA_ROOT` variable to the root at which you are storing the images
- Download the `cityImages.zip` from this [Google Drive](https://drive.google.com/drive/folders/1MbULVSayy85VhRgYFyU1QzC6GpDkCJ-f?usp=sharing)
    - Make the location of `cityImages` the root of the `DATA_ROOT` variable
    - These images are taken from the [Mapillary Street-level Sequences Dataset](https://www.mapillary.com/datasets)
- Run `main.py` in this directory

## Model 2: Linear Regression
- Install the `image_embeddings.npy` from this [Google Drive](https://drive.google.com/drive/folders/1MbULVSayy85VhRgYFyU1QzC6GpDkCJ-f?usp=sharing)
    - This file stores the image embeddings for city images in the [GSV-cities](https://www.kaggle.com/datasets/amaralibey/gsv-cities) dataset using [Google's EfficientNet-b5](https://research.google/blog/efficientnet-improving-accuracy-and-efficiency-through-automl-and-model-scaling/) hosted on [HuggingFace](https://huggingface.co/google/efficientnet-b5)

- Go to `src/embeddings`
- Move the `image_embeddings.npy` file to this directory
- Go to `src/embeddings/runModel`
    - Change `RUN_LINREG` to `True` in `main.py`
    - Change `RUN_NN` to `False` in `main.py`
    - Change `IMAGE_PATH` to the path of the image you want to predict the location of
- Run `main.py` in this directory

## Model 3: Neural Net
- Make sure to have the `image_embeddings.npy` file in the `src/embeddings` directory, see install instructions above
- Go to `src/embeddings`
- Move the `image_embeddings.npy` file to this directory
- Go to `src/embeddings/runModel`
    - Change `RUN_LINREG` to `False` in `main.py`
    - Change `RUN_NN` to `True` in `main.py`
    - Change `IMAGE_PATH` to the path of the image you want to predict the location of
- Run `main.py` in this directory

## Other Data
- To visualize US Cities, we used data from the [US Cities Database](https://simplemaps.com/data/us-cities)