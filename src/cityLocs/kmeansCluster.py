from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull

# TODO: Make sure to cite this source if using these cities
# https://simplemaps.com/data/us-cities
us_cities = pd.read_csv("uscities.csv")
us_cities = us_cities.dropna()

# Set seed for reproducibility
# np.random.seed(10)

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
PERCENT_CITIES = 1
continental_us_cities = continental_us_cities.sample(frac=PERCENT_CITIES)

print(f"Number of Cities Total: {len(continental_us_cities)}")

# Specify the number of clusters
k = 15

# Perform K-means clustering
kmeans = KMeans(n_clusters=k)
kmeans.fit(continental_us_cities[['lng', 'lat']])

# Get cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the data
fig, ax = plt.subplots()
plt.scatter(continental_us_cities['lng'],
            continental_us_cities['lat'], c=labels, s=2)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=10)

# Draw bounding polygons around clusters
for i in range(k):
    cluster_points = continental_us_cities.loc[labels == i, [
        'lng', 'lat']].values
    hull = ConvexHull(cluster_points)
    plt.plot(np.append(cluster_points[hull.vertices, 0], cluster_points[hull.vertices[0], 0]),
             np.append(cluster_points[hull.vertices, 1], cluster_points[hull.vertices[0], 1]), 'k-', lw=1)

plt.xlabel('Longitude')
plt.ylabel('Latitute')
plt.title('US Cities K-means Clustering')
plt.show()
