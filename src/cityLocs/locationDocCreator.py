import pandas as pd
import os
from tqdm import tqdm

GSV_IMAGE_ROOT = "D:\WPI\Junior Year\ML\CS539_Project\data\ImageLocationDataset\GSV_USCities"

locations_df = pd.DataFrame(columns=['city', 'lat', 'lng'])

for city in os.listdir(GSV_IMAGE_ROOT):
    print(city)
    city_path = os.path.join(GSV_IMAGE_ROOT, city)
    for image in tqdm(os.listdir(city_path)):
        city = image.split("_")[0]
        lat = float(image.split("_")[5])
        lng = float(image.split("_")[6])
        locations_df = pd.concat([locations_df, pd.DataFrame([[city, lat, lng]], columns=['city', 'lat', 'lng'])])

locations_df.to_csv("gsv_image_locations.csv", index=False)
