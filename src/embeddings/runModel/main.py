import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import pickle

RUN_NN = True
RUN_LINREG = False

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

IMAGE_PATH = "C:/Users/bugsp/Downloads/boston1.jpg"
# Generate image embedding for input
print("Generating image embedding")
image = Image.open(IMAGE_PATH)
image_input = processor(image, return_tensors="pt")
image_input = {k: v.to(device) for k, v in image_input.items()}
with torch.no_grad():
    image_embedding = image_encoder(**image_input).logits.cpu().numpy()


if RUN_LINREG:
    # Load linear regression model to predict GPS coordinates from image embeddings
    print("Loading linear regression model")
    with open('../regression_model.pkl', 'rb') as f:
        linreg = pickle.load(f)

    # Predict GPS coordinates from image embedding
    print("Predicting GPS coordinates")
    predicted_coords = linreg.predict(image_embedding)

if RUN_NN:
    print("Loading neural network model")
    nn_scaler = torch.load('../nn_scaler.pth')
    model = GPSPredictor(input_dim=1000, output_dim=1)
    model.load_state_dict(torch.load('../gps_predictor.pth'))
    model.to(device)
    model.eval()

    # Predict GPS coordinates from image embedding
    print("Predicting GPS coordinates")
    predicted_coords_tensor = model(torch.tensor(image_embedding, dtype=torch.float32).to(device))
    predicted_coords = nn_scaler.inverse_transform(predicted_coords_tensor.detach().cpu().numpy())

def decode_gps(embedded_coords):
    decoded_coord_tensor = coord_encoder.decoder(torch.tensor(
        embedded_coords, dtype=torch.float32).to(device)).detach().cpu().numpy()
    decoded_coords = scaler.inverse_transform(decoded_coord_tensor)
    return decoded_coords


predicted_gps = decode_gps(predicted_coords)
print(f"Predicted GPS coordinates: {predicted_gps}")
