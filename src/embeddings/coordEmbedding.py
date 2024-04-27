import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Define the Autoencoder architecture


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1200),
            nn.ReLU(),
            nn.Linear(1200, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1500),
            nn.ReLU(),
            nn.Linear(1500, 2000),
            nn.ReLU(),
            nn.Linear(2000, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

COORD_DATA_ROOT = "D:/WPI/Junior Year/ML/CS539_Project/src/cityLocs"
FILE_NAME = "gsv_image_locations.csv"

df = pd.read_csv(os.path.join(COORD_DATA_ROOT, FILE_NAME))
latitude = df['lat']
longitude = df['lng']

print(latitude.size)

input_data = np.column_stack((latitude, longitude))

# Normalize input data (optional but recommended)
input_data = (input_data - np.mean(input_data, axis=0)) / \
    np.std(input_data, axis=0)

# Convert numpy arrays to PyTorch tensors and move to GPU if available
input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

# Set hyperparameters
input_dim = 2  # Dimensionality of input data (latitude and longitude pairs)
latent_dim = 1000  # Dimensionality of latent space representation
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Instantiate the autoencoder model and move to GPU if available
model = Autoencoder(input_dim, latent_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Shuffle input data
    indices = torch.randperm(input_tensor.size(0))
    shuffled_input = input_tensor[indices]

    # Mini-batch training
    for i in tqdm(range(0, len(input_tensor), batch_size)):
        batch_input = shuffled_input[i:i + batch_size]

        # Move batch to GPU if available
        batch_input = batch_input.to(device)

        # Forward pass
        reconstructed = model(batch_input)

        # Compute loss
        loss = criterion(reconstructed, batch_input)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss after each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Encode input data into the latent space
with torch.no_grad():
    encoded_data = model.encoder(
        input_tensor).cpu().numpy()  # Move data back to CPU

# Save embeddings to a file
np.save('encoded_lat_lon.npy', encoded_data)
