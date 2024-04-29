
import torch
from skimage import io
from torch.utils.data import Dataset
import os

DATA_ROOT = "D:/WPI/Junior Year/ML/CS539_Project/data/cityImages"

class CityImageSet(Dataset):
    """
    - Collections of images from 6 cities: Boston, Amsterdam, Paris, Phoenix, Toronto, Zurich
    - All images are 250x250 pixels
    """

    def __init__(self, city):
        self.city = city
        self.images = [img for img in os.listdir(
            f"{DATA_ROOT}/{city}/output_images")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(
            f"{DATA_ROOT}/{self.city}/output_images/{self.images[idx]}")
        # Convert to float tensor
        image = torch.tensor(image, dtype=torch.float)
        # Transpose image dimensions to match PyTorch convention: [channels, height, width]
        image = image.permute(2, 0, 1)
        label = 0 if self.city == "boston" else 1 if self.city == "amsterdam" else 2 if self.city == "paris" else 3 if self.city == "phoenix" else 4 if self.city == "toronto" else 5
        return {'image': image, 'city': label}
