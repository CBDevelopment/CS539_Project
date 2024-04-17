
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
print(len(us_cities))

plt.scatter(us_cities['lng'], us_cities['lat'])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("US Cities")
plt.show()