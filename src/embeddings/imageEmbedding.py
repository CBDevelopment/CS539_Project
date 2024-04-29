import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# Load model directly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
processor = AutoImageProcessor.from_pretrained("google/efficientnet-b5")
model = AutoModelForImageClassification.from_pretrained(
    "google/efficientnet-b5").to(device)

# Define a function to generate embeddings


def generate_embeddings(image_paths, batch_size=32):
    embeddings = []
    num_batches = len(image_paths) // batch_size + 1

    for i in tqdm(range(0, num_batches)):  # Process batches one by one
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        # Load images using PIL
        batch_images = [Image.open(path) for path in batch_paths]
        batch_inputs = processor(batch_images, return_tensors="pt")

        # Move inputs to GPU
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.logits
            # Extract logits and convert to numpy array
            embeddings.extend(logits.cpu().numpy())

    return np.array(embeddings)


# Set the path to your image directory
# GSV_IMAGE_ROOT = "D:/WPI/Junior Year/ML/CS539_Project/data/ImageLocationDataset/GSV_USCities"

GSV_IMAGE_ROOT = "/home/cjbeck/cs539/TuringData"
image_paths = []
for city in os.listdir(GSV_IMAGE_ROOT):
    city_path = os.path.join(GSV_IMAGE_ROOT, city)
    for image in os.listdir(city_path):
        image_paths.append(os.path.join(city_path, image))

# Example usage
image_embeddings = generate_embeddings(image_paths)
print(image_embeddings.shape)

# Save embeddings to a file
np.save('image_embeddings.npy', image_embeddings)
