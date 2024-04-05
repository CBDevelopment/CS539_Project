# train.py

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import output_constants as io 

## HYPER PARAMETERS ####################################
NUM_EPOCHS      = 25
ALPHA           = 0.001
BATCH_SIZE      = 64

DATASET = "data_5drivers/*.csv"
########################################################

class TaxiDriverDataset(Dataset):
    """
    Handles loading and preparing data for the model
    """
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model, optimizer, criterion, train_loader, device):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """
    model.train()
    train_loss = 0.0
    correct_train = 0

    for inputs, labels in tqdm(train_loader, desc=f"{io.BLUE_TEXT}Training{io.RESET_TEXT}"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()

    # Compute average training loss and accuracy
    avg_train_loss  = train_loss    / len(train_loader)
    train_acc       = correct_train / len(train_loader.dataset)

    return avg_train_loss, train_acc

def evaluate(model, criterion, test_loader, device):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    """
    model.eval()
    test_loss = 0.0
    correct_test = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"{io.BLUE_TEXT}Validating{io.RESET_TEXT}"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()

    # Compute average validation loss and accuracy
    avg_test_loss   = test_loss     / len(test_loader)
    test_acc        = correct_test  / len(test_loader.dataset)

    return avg_test_loss, test_acc

def train_model(model_number, debug=False):
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """
    print(f"TRAINING... ")
    print(f"\tmodel_number - {model_number}")
    print(f"\tdebug        - {debug}")
    DEBUG = debug
    if (DEBUG):
        print(f"\t{io.BLUE_TEXT}DEBUG STATUS:{io.RESET_TEXT} {io.GREEN_TEXT}{DEBUG}{io.RESET_TEXT}")
    else:
        print(f"\t{io.BLUE_TEXT}DEBUG STATUS:{io.RESET_TEXT} {io.YELLOW_TEXT}{DEBUG}{io.RESET_TEXT}")
    if (DEBUG):
        print(f"\tSince DEBUG is enabled, this program will have {io.GREEN_TEXT}enhanced{io.RESET_TEXT} output!")
    print(f"TRAINING complete... \n")
