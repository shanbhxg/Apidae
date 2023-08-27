import numpy as np

class CosineKMeans:
    def __init__(self, num_clusters, max_iters=100):
        self.num_clusters = num_clusters
        self.max_iters = max_iters

    def fit(self, data):
        self.data = data
        self.num_samples, self.num_features = data.shape
        self.centroids = self.data[np.random.choice(self.num_samples, self.num_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid based on cosine similarity
            distances = self._cosine_similarity(self.data, self.centroids)
            labels = np.argmax(distances, axis=1)

            # Update centroids based on the mean of assigned data points
            new_centroids = np.array([self.data[labels == i].mean(axis=0) for i in range(self.num_clusters)])

            # Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids

        self.labels = labels
        self.cluster_centers = self.centroids

    def predict(self, data):
        distances = self._cosine_similarity(data, self.cluster_centers)
        labels = np.argmax(distances, axis=1)
        return labels

    def _cosine_similarity(self, a, b):
        dot_product = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        cosine_similarity = dot_product / (norm_a * norm_b.T)
        return cosine_similarity

# # Example usage
# if __name__ == "__main__":
#     np.random.seed(42)
    
#     # Generate synthetic data
#     num_samples = 100
#     num_features = 10
#     data = np.random.rand(num_samples, num_features)
    
#     # Create and fit the CosineKMeans model
#     num_clusters = 3
#     kmeans = CosineKMeans(num_clusters=num_clusters)
#     kmeans.fit(data)
    
#     # Predict cluster labels for new data
#     new_data = np.random.rand(5, num_features)
#     predicted_labels = kmeans.predict(new_data)
#     print("Predicted labels:", predicted_labels)
