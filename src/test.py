# test.py

import torch
from torch.utils.data import DataLoader

import model as model_definitions
import main

from extract_feature import load_data, preprocess_data
from train import ImageSet

import output_constants as io 

## PARAMETERS ####################################
MODEL_PATH      = "model.pt"

NUM_EPOCHS      = main.NUM_EPOCHS
ALPHA           = main.ALPHA
BATCH_SIZE      = main.BATCH_SIZE
##################################################

# TODO:
def test(model, test_loader, device):
    """
    Test the model performance on the test set.
    """
    model.eval()
    test_loss = 0
    correct_predictions = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = correct_predictions / len(test_loader.dataset)

    return avg_loss, accuracy

# TODO:
def test_model(model_number, debug=False, seed=42):
    """
    Main function to initiate the model testing process.
    Includes loading test data, setting up the model and test loader,
    and executing the testing function.
    """
    print(f"TESTING... ")
    DEBUG       = debug
    SET_SEED    = seed

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

    # Load test data
    test_file_pattern = 'data_exp/*.csv'
    X_test, y_test = load_data(test_file_pattern)

    # Preprocess test data
    test_dataset    = ImageSet(X_test, y_test, device)
    test_loader     = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_test.shape[2]     # Adjust based on input feature dimensions
    output_dim = len(set(y_test))   # Number of classes
    
    # Initialize and load the model
    model = -1
    
    # Define which model architecture being used
    match (model_number):
        case 1:
            model = model_definitions.Boston_NYC_BinaryClassifier(input_dim, output_dim).to(device)        
        case _:
            print("Reached (what *should* be) an unreachable state")
            exit(1)

    # Load the trained model's weights
    model.load_state_dict(torch.load(f"{MODEL_PATH}", map_location=device))

    # Test the model
    test_loss, accuracy = test(model, test_loader, device)

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
