import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import pickle

# Importing was being a pain so I just copied it here
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

print("Initializing device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load image encoder
print("Loading image encoder")
processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
image_encoder = AutoModelForImageClassification.from_pretrained(
    "google/efficientnet-b5").to(device)

# Load GPS encoder
print("Loading GPS encoder")
coord_encoder = Autoencoder(input_dim=2, latent_dim=1)
coord_encoder.load_state_dict(torch.load('../autoencoder.pth'))
coord_encoder.to(device)
coord_encoder.eval()
scaler = torch.load('../scaler.pth')

# Load linear regression model to predict GPS coordinates from image embeddings
print("Loading linear regression model")
with open('../regression_model.pkl', 'rb') as f:
    linreg = pickle.load(f)

IMAGE_PATH = "D:\WPI\Junior Year\ML\CS539_Project\data\ImageLocationDataset\TuringData\Boston\Boston_0000001_2007_09_119_42.32195180265299_-71.1349467783536_k6Bs7PgCwPKNLkiPKyBx3A.jpg"
# Generate image embedding for input
print("Generating image embedding")
image = Image.open(IMAGE_PATH)
image_input = processor(image, return_tensors="pt")
image_input = {k: v.to(device) for k, v in image_input.items()}
with torch.no_grad():
    image_embedding = image_encoder(**image_input).logits.cpu().numpy()

# Predict GPS coordinates from image embedding
print("Predicting GPS coordinates")
predicted_coords = linreg.predict(image_embedding)


def decode_gps(embedded_coords):
    decoded_coord_tensor = coord_encoder.decoder(torch.tensor(
        embedded_coords, dtype=torch.float32).to(device)).detach().cpu().numpy()
    decoded_coords = scaler.inverse_transform(decoded_coord_tensor)
    return decoded_coords


predicted_gps = decode_gps(predicted_coords)
print(f"Predicted GPS coordinates: {predicted_gps}")
