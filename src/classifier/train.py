
import torch
import torch.nn as nn
import numpy as np
from classifier import Classifier

from transformers import AutoImageProcessor, AutoModelForImageClassification

print("Initializing device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load image embeddings and GPS coordinates
print("Loading data...")
boston_embeddings = np.load('../embeddings/imageEmbeddings/Boston_embeddings.npy', allow_pickle=True)
chicago_embeddings = np.load('../embeddings/imageEmbeddings/Chicago_embeddings.npy', allow_pickle=True)
minneapolis_embeddings = np.load('../embeddings/imageEmbeddings/Minneapolis_embeddings.npy', allow_pickle=True)
washdc_embeddings = np.load('../embeddings/imageEmbeddings/WashingtonDC_embeddings.npy', allow_pickle=True)

classifier = Classifier(input_dim=1000, output_dim=4)

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

def train(model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

