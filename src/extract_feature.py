import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from tqdm import tqdm

# TODO:
    # load_data() is standard boiler plate code 
        # It needs to be adapted from CSV to an image file pattern 
def load_data(file_pattern):
    """
    Load and process image files matching the given file pattern, one at a time.
    """
    all_files = glob.glob(file_pattern)
    processed_data = []

    with tqdm(total=len(all_files), desc="Processing files") as pbar:
        for filename in all_files:
            # TODO:
                # Change from .csv to image file pattern
            df = pd.read_csv(filename, index_col=None, header=0)
            X_processed, y_processed = preprocess_data(df)
            if X_processed.size > 0:  # Check if there's data to add
                processed_data.append((X_processed, y_processed))
            pbar.update(1)

    # Combine data from all files
    X_combined = np.concatenate([data[0] for data in processed_data], axis=0) if processed_data else np.array([])
    y_combined = np.concatenate([data[1] for data in processed_data], axis=0) if processed_data else np.array([])

    return X_combined, y_combined

# TODO:
def preprocess_data(frame):
    """
    Perform data preprocessing: 
        feature extraction  : Pixel --> 3d tensor [R(n x n), G(n x n), B(n x n)], class (0, 1, ..., n)
        scaling             : values (0, 255) --> (0, 1)
        reshaping           : data shaped for model
    """

    return X_reshaped, y_reshaped