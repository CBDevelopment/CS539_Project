# main.py

import sys
import output_constants as io 
from model import NUM_MODELS
from train import train_model
#from test import test_model

DEBUG           = True
SET_SEED        = 42
NUM_EPOCHS      = 25
ALPHA           = 0.001
BATCH_SIZE      = 64

# #####################################
#
# Run me with 'py main.py <train/test> <1>'
#
# #####################################

def main():
    # Check if the user has provided an argument (train or test) (model = [1, NUM_MODELS)])
    if len(sys.argv) != 3:
        print(f"{io.RED_TEXT}\tUsage: py main.py <mode> <model>{io.RESET_TEXT}")
        print(f"\t\t{io.CYAN_TEXT}<mode> should be 'train' or 'test'{io.RESET_TEXT}")
        print(f"\t\t{io.MAGENTA_TEXT}<model number> should be between [1, {NUM_MODELS}){io.RESET_TEXT}\n")
        sys.exit(1)  # Exit the program if no argument is provided

    # Fetch the mode from the command line argument
    mode = sys.argv[1]

    # Fetch the model number from the command line argument
    model_number = int(sys.argv[2])

    if mode != 'train' and mode != 'test':
        print(f"\t{io.RED_TEXT}Invalid mode. Received {io.YELLOW_TEXT}{mode}{io.RED_TEXT}. Choose 'train' or 'test'{io.RESET_TEXT}\n")
        sys.exit(1)  # Exit if an invalid mode is provided

    if model_number not in range(1, NUM_MODELS + 1):
        print(f"\t{io.RED_TEXT}Invalid model. Received {io.YELLOW_TEXT}{model_number}{io.RED_TEXT}. Choose between [1, {NUM_MODELS}){io.RESET_TEXT}\n")
        sys.exit(1)  # Exit if an invalid model is provided

    if mode == 'train':
        print("train")
        train_model(model_number, DEBUG, SET_SEED)  
    else:
        print("test")
        #test_model(model_number, DEBUG, SET_SEED)  
        
if __name__ == '__main__':
    main()
