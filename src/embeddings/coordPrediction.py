from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
import torch

# Load image embeddings and GPS coordinates
print("Loading data...")
image_embeddings = np.load('image_embeddings.npy')
encoded_coords = np.load('encoded_lat_lon.npy')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(image_embeddings, encoded_coords, test_size=0.2, random_state=42)

print("Training model")
# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
val_predictions = model.predict(X_val)
val_loss = np.mean(np.square(val_predictions - y_val))  # Mean squared error
print(f"Validation Loss: {val_loss:.4f}")

print("Visualizing predictions")
coord_encoder = Autoencoder(input_dim=2, latent_dim=1)
coord_encoder.load_state_dict(torch.load('autoencoder.pth'))
scaler = torch.load('scaler.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
coord_encoder.to(device)
coord_encoder.eval()    

def decode_gps(embedded_coords):
    decoded_coord_tensor = coord_encoder.decoder(torch.tensor(embedded_coords, dtype=torch.float32).to(device)).detach().cpu().numpy()
    decoded_coords = scaler.inverse_transform(decoded_coord_tensor)
    return decoded_coords

predicted_gps = decode_gps(val_predictions)
actual_gps = decode_gps(y_val)

# Plot the predicted and actual GPS coordinates on 2 axes side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(actual_gps[:, 0], actual_gps[:, 1], label='Actual GPS Coordinates')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Actual GPS Coordinates')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(predicted_gps[:, 0], predicted_gps[:, 1], label='Predicted GPS Coordinates')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Predicted GPS Coordinates')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
torch.save(model, 'regression_model.pth')
