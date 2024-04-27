import numpy as np
from sklearn.linear_model import LinearRegression

# Load labeled dataset of paired image embeddings and GPS coordinates
image_embeddings = np.load('image_embeddings.npy')
encoded_coords = np.load('encoded_lat_lon.npy')

# Train a linear regression model to predict GPS coordinates from image embeddings
model = LinearRegression()
model.fit(image_embeddings, encoded_coords)

# Example usage: predict GPS coordinate pair for a query image embedding
query_image_embedding = np.random.rand(1, 1000)  # Example query image embedding
predicted_gps_coordinate = model.predict(query_image_embedding)
print("Predicted GPS Coordinate Pair:", predicted_gps_coordinate)
