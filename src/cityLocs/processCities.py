
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d

# # https://simplemaps.com/data/world-cities
# # TODO: Make sure to cite this source if using it
# cities = pd.read_csv("worldcities.csv")
# cities = cities.dropna()
# us_cities = cities[cities['iso2'] == 'US']
# print(len(us_cities))

# https://simplemaps.com/data/us-cities
# TODO: reference this source if using
us_cities = pd.read_csv("uscities.csv")
us_cities = us_cities.dropna()
continental_us_cities = us_cities[(us_cities['lat'] > 24.396308) & (us_cities['lat'] < 49.384358) & (us_cities['lng'] > -125.0) & (us_cities['lng'] < -66.93457)]

continental_us_cities = continental_us_cities.sample(frac=0.01)
print(len(continental_us_cities))

# vor = Voronoi(continental_us_cities[['lng', 'lat']])
# voronoi_plot_2d(vor)
# plt.xlim(-125, -66)
# plt.ylim(24, 49)
# plt.show()

plt.scatter(continental_us_cities['lng'], continental_us_cities['lat'], s=1)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("US Cities")
plt.show()