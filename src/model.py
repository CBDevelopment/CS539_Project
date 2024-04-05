# model.py

import torch
import torch.nn as nn

NUM_MODELS = 2

'''
    Initial Model architecture

    Binary Image Classifier:
        Class 0:    Boston
        Class 1:    New York City
'''
class Boston_NYC_BinaryClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Boston_NYC_BinaryClassifier, self).__init__()
        
        # Model metadata
        ## Good for I/O & Internal Data Handling
        self.model_class = f"Boston_NYC_BinaryClassifier"

    def forward(self, x):

        return x