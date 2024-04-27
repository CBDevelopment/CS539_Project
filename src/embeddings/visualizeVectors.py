import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the embeddings
vectors = np.load('encoded_lat_lon.npy')

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Plot the reduced vectors
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], s=10)
plt.title('Visualization of 1000-dimensional vectors (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
