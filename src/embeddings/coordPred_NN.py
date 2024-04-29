from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load image embeddings and GPS coordinates
print("Loading data...")
image_embeddings = np.load('image_embeddings.npy')
encoded_coords = np.load('encoded_lat_lon.npy')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(image_embeddings, encoded_coords, test_size=0.2, random_state=42)

# normalize all data to be between 0 and 1
scaler = MinMaxScaler()
scaler.fit(y_train)

# build neural net
class GPSPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GPSPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, 300)
        self.fc4 = nn.Linear(300, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
    

# train predictor
print("Training model")
model = GPSPredictor(input_dim=1000, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(scaler.transform(y_train), dtype=torch.float32).to(device)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = scaler.transform(y_val)

val_predictions = model(X_val_tensor).detach().cpu().numpy()
print(val_predictions)
val_loss = np.mean(np.square(val_predictions - y_val))  # Mean squared error
print(f"Validation Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'gps_predictor.pth')

autoencoder = Autoencoder(input_dim=2, latent_dim=1)
autoencoder.load_state_dict(torch.load('autoencoder.pth'))
autoencoder.to(device)

def decode_gps(embedded_coords):
    encoder_scaler = torch.load('scaler.pth')
    decoded_coord_tensor = autoencoder.decoder(torch.tensor(embedded_coords, dtype=torch.float32).to(device)).detach().cpu().numpy()
    decoded_coords = encoder_scaler.inverse_transform(decoded_coord_tensor)
    return decoded_coords

val_predictions = scaler.inverse_transform(val_predictions)
y_val = scaler.inverse_transform(y_val)

predicted_gps = decode_gps(val_predictions)
actual_gps = decode_gps(y_val)

# Plot the predicted and actual GPS coordinates on 2 axes side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(actual_gps[:, 1], actual_gps[:, 0], label='Actual GPS Coordinates')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Actual GPS Coordinates')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(predicted_gps[:, 1], predicted_gps[:, 0], label='Predicted GPS Coordinates')
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.title('Predicted GPS Coordinates')
plt.legend()

plt.tight_layout()
plt.show()
