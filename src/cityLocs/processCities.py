
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm
import math
from sklearn.cluster import KMeans

us_cities = pd.read_csv("uscities.csv")
us_cities = us_cities.dropna()

# Set seed for reproducibility
np.random.seed(10)

# Continental US Bounding Box
north = 49.3457868
south = 24.7433195
west = -124.7844079
east = -66.9513812

BOUND_NORTHEAST = True
if BOUND_NORTHEAST:
    # Bounding Box for "Northeast" US
    north = 49
    south = 34.6
    west = -98.8
    east = -66.9

continental_us_cities = us_cities[
    (us_cities['lat'] > south)
    & (us_cities['lat'] < north)
    & (us_cities['lng'] > west)
    & (us_cities['lng'] < east)
]

# Percentage of cities to plot, decimal
PERCENT_CITIES = 0.05
continental_us_cities = continental_us_cities.sample(frac=PERCENT_CITIES)

print(f"Number of Cities Total: {len(continental_us_cities)}")

fig, ax = plt.subplots()

# Plot image locations from the GSV Data set
gsv_img_locs = pd.read_csv("gsv_image_locations.csv")
gsv_img_locs = gsv_img_locs[
    (gsv_img_locs['lat'] > south)
    & (gsv_img_locs['lat'] < north)
    & (gsv_img_locs['lng'] > west)
    & (gsv_img_locs['lng'] < east)
]
plt.scatter(gsv_img_locs['lng'], gsv_img_locs['lat'], s=2, c='r')

gsv_included_cities = gsv_img_locs['city'].unique()
print(f"Number of Cities in GSV Dataset: {len(gsv_included_cities)}")

city_centers = pd.DataFrame(columns=['city', 'lat', 'lng'])
for city in gsv_included_cities:
    city_imgs = gsv_img_locs[gsv_img_locs['city'] == city]
    city_avg_lat = city_imgs['lat'].mean()
    city_avg_lng = city_imgs['lng'].mean()
    if ((city_avg_lat > south and city_avg_lat < north) and (city_avg_lng > west and city_avg_lng < east)):
        city_centers = pd.concat([city_centers, pd.DataFrame(
            [[city, city_avg_lat, city_avg_lng]], columns=['city', 'lat', 'lng'])])
        plt.scatter(city_avg_lng, city_avg_lat, s=10, c='purple')

LAT_TO_MILES = 69  # 69 miles per degree of latitude
LNG_TO_MILES = 54.6  # 54.6 miles per degree of longitude

PLOT_RADIUS = False

# Only plot US Cities that are within a certain radius of any city's average center
RADIUS = 400  # miles

if PLOT_RADIUS:
    # Cities to plot
    cities_to_plot = []
    for i, city in tqdm(continental_us_cities.iterrows()):
        city_lat = city['lat']
        city_lng = city['lng']

        for i, city_center in city_centers.iterrows():
            center_lat = city_center['lat']
            center_lng = city_center['lng']

            # find cities that are in a circle of radius RADIUS
            # distance between two points = sqrt((x2 - x1)^2 + (y2 - y1)^2)
            distance = math.sqrt((abs(center_lat - city_lat) * LAT_TO_MILES)
                                 ** 2 + (abs(center_lng - city_lng) * LNG_TO_MILES) ** 2)

            if distance < RADIUS:
                cities_to_plot.append(city)
                break

    cities_to_plot = pd.DataFrame(cities_to_plot)
    print(f"Number of Cities to Plot: {len(cities_to_plot)}")
else:
    cities_to_plot = continental_us_cities

# Whether to plot just cities or Voronoi diagram
VORONOI = True
PLOT_CITIES = True

if VORONOI:
    vor = Voronoi(cities_to_plot[['lng', 'lat']])
    voronoi_plot_2d(vor, ax=ax, line_width=0.5, show_vertices=False,
                    show_points=PLOT_CITIES, point_size=1)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("US Cities Voronoi Diagram")
else:
    plt.scatter(cities_to_plot['lng'],
                cities_to_plot['lat'], s=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("US Cities")

plt.show()
