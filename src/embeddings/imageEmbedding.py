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

    for i in tqdm(range(0, num_batches, torch.cuda.device_count())):  # Process multiple batches simultaneously
        batch_paths = image_paths[i * batch_size: (i + torch.cuda.device_count()) * batch_size]
        batch_images = [Image.open(path) for path in batch_paths]  # Load images using PIL
        batch_filenames = np.array([os.path.basename(path) for path in batch_paths])  # Extract filenames

        batch_inputs = processor(batch_images, return_tensors="pt")

        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}  # Move inputs to GPU

        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.logits
            batch_embeddings = logits.cpu().numpy()  # Extract logits and convert to numpy array

            # Combine filenames with embeddings
            for filename, embedding in zip(batch_filenames, batch_embeddings):
                embedding = embedding.reshape(-1)
                embeddings.append([filename, embedding.tolist()])
    
    embeddings_array = np.array(embeddings, dtype=object)

    return embeddings_array


# Set the path to your image directory
# GSV_IMAGE_ROOT = "D:/WPI/Junior Year/ML/CS539_Project/data/ImageLocationDataset/GSV_USCities"

GSV_IMAGE_ROOT = "D:\WPI\Junior Year\ML\CS539_Project\data\ImageLocationDataset\TuringData"

# for city in os.listdir(GSV_IMAGE_ROOT):
for city in ["Chicago", "Minneapolis", "WashingtonDC"]:
    print(city)
    city_path = os.path.join(GSV_IMAGE_ROOT, city)
    image_paths = []
    for image in os.listdir(city_path):
        image_paths.append(os.path.join(city_path, image))

    image_embeddings = generate_embeddings(image_paths)

    np.save(f'imageEmbeddings/{city}_embeddings.npy', image_embeddings)
