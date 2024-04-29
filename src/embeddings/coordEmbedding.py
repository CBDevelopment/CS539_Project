import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from autoencoder import Autoencoder


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

COORD_DATA_ROOT = "D:/WPI/Junior Year/ML/CS539_Project/src/cityLocs"
FILE_NAME = "NE_gsv_image_locations.csv"

df = pd.read_csv(os.path.join(COORD_DATA_ROOT, FILE_NAME))
latitude = df['lat']
longitude = df['lng']

input_data = np.column_stack((latitude, longitude))

# Scale input data to the range [0, 1]
scaler = MinMaxScaler()
scaled_input_data = scaler.fit_transform(input_data)

# Convert numpy arrays to PyTorch tensors and move to GPU if available
input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32).to(device)

# Set hyperparameters
input_dim = 2  # Dimensionality of input data (latitude and longitude pairs)
latent_dim = 1  # Dimensionality of latent space representation
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Initialize the autoencoder
autoencoder = Autoencoder(input_dim, latent_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = autoencoder(input_tensor)
    loss = criterion(outputs, input_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Encode the GPS coordinates
encoded_coords_tensor = autoencoder.encoder(
    input_tensor).detach().cpu().numpy()

# Decode the encoded representation to reconstruct the GPS coordinates
decoded_coords_tensor = autoencoder.decoder(torch.tensor(
    encoded_coords_tensor, dtype=torch.float32).to(device)).detach().cpu().numpy()

# Inverse transform to get the original scale
decoded_coords = scaler.inverse_transform(decoded_coords_tensor)

# Print the original, encoded, and decoded GPS coordinates
print("\nOriginal GPS Coordinates:")
print(input_data)
print("\nEncoded GPS Coordinates:")
print(encoded_coords_tensor)
print("\nDecoded GPS Coordinates:")
print(decoded_coords)

# Save embeddings to a file
np.save('encoded_lat_lon.npy', encoded_coords_tensor)

# Save model and scaler
torch.save(autoencoder.state_dict(), 'autoencoder.pth')
torch.save(scaler, 'scaler.pth')
