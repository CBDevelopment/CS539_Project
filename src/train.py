# train.py

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import output_constants as io 
import main 
import model as model_definitions
from extract_feature import load_data

## HYPER PARAMETERS ####################################
NUM_EPOCHS      = main.NUM_EPOCHS
ALPHA           = main.ALPHA
BATCH_SIZE      = main.BATCH_SIZE

TEST_SIZE       = 0.2

DATA    = "boston_and_nyc"
DATASET = f"training_images/{DATA}/*.png"
########################################################

# TODO: 
class ImageSet(Dataset):
    """
    Handles loading and preparing data for the model
    """
    def __init__(self, X, y, device):
        # TODO: self.X is likely to change because of how we handle/encode pixel data
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO:
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

# TODO:
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

# TODO:
def train_model(model_number, debug=False, seed=42):
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """
    print(f"TRAINING... ")
    DEBUG = debug
    SET_SEED = seed
    if (DEBUG):
        print(f"\t{io.BLUE_TEXT}DEBUG STATUS:{io.RESET_TEXT} {io.GREEN_TEXT}{DEBUG}{io.RESET_TEXT}")
        print(f"\t{io.BLUE_TEXT}MODEL NUMBER:{io.RESET_TEXT} {io.GREEN_TEXT}{model_number}{io.RESET_TEXT}")
        print(f"\t{io.BLUE_TEXT}Since DEBUG is enabled, this program will have {io.GREEN_TEXT}enhanced{io.BLUE_TEXT} output!{io.RESET_TEXT}")
    else:
        print(f"\t{io.BLUE_TEXT}DEBUG STATUS:{io.RESET_TEXT} {io.YELLOW_TEXT}{DEBUG}{io.RESET_TEXT}")
    
    ###########################################################################################################################################
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if (DEBUG):
            print(f"{io.MAGENTA_TEXT}\t== GPU Details {'='*30}{io.RESET_TEXT}")
            print(f"\t{io.MAGENTA_TEXT}ǁ Name{io.RESET_TEXT}            | {io.GREEN_TEXT}{torch.cuda.get_device_name(0)}{io.RESET_TEXT}")
            print(f"\t{io.MAGENTA_TEXT}ǁ Total Memory{io.RESET_TEXT}    | {io.GREEN_TEXT}{format(torch.cuda.get_device_properties(0).total_memory / 1024**3, '2.4f')} GB{io.RESET_TEXT}")
            print(f"\t{io.MAGENTA_TEXT}ǁ Multiprocessors{io.RESET_TEXT} | {io.GREEN_TEXT}{torch.cuda.get_device_properties(0).multi_processor_count}{io.RESET_TEXT}")
            print(f"{io.MAGENTA_TEXT}\t==============={'='*30}{io.RESET_TEXT}")    
        torch.cuda.manual_seed(SET_SEED)
        print(f"\tGPU with seed set to {io.BLUE_TEXT}{SET_SEED}{io.RESET_TEXT}")
    else:
        device = torch.device("cpu")
        torch.manual_seed(SET_SEED)
        print("\tCPU")
        print(f"\tCPU with seed set to {io.BLUE_TEXT}{SET_SEED}{io.RESET_TEXT}")
    ###########################################################################################################################################

    # Load and preprocess training data
    X, y = load_data(DATASET)

    # Split training data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    train_dataset       = ImageSet(X_train, y_train, device)
    validate_dataset    = ImageSet(X_test, y_test, device)
    train_loader        = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    validate_loader     = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False) 
    
    input_dim = X_test.shape[2]     # Adjust based on input feature dimensions
    output_dim = len(set(y_test))   # Number of classes
        
    # Initialize the model
    model = -1

    # Define which model architecture being used
    match (model_number):
        case 1:
            model = model_definitions.Boston_NYC_BinaryClassifier(input_dim, output_dim).to(device)        
        case _:
            print("Reached (what *should* be) an unreachable state")
            exit(1)

    # TODO:
    # Define the training loop

    ## For epochs (0, NUM_EPOCHS)
    ### Train()
    ### Evaluate()
    
    print(f"TRAINING complete... \n")
