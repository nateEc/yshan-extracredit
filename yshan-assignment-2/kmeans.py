import numpy as np

class KMeans:
    def __init__(self, k, init_method="random"):
        self.k = k
        self.init_method = init_method
        self.centroids = None

    def initialize_centroids(self, data):
        if self.init_method == "random":
            self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        elif self.init_method == "farthest_first":
            # Farthest first initialization logic here
            pass
        elif self.init_method == "kmeans++":
            # KMeans++ initialization logic here
            pass
        elif self.init_method == "manual":
            # Manual initialization will be handled from the frontend.
            pass
        else:
            raise ValueError("Unknown initialization method")
    
    def assign_points_to_clusters(self, data):
        clusters = []
        for point in data:
            distances = np.linalg.norm(point - self.centroids, axis=1)
            clusters.append(np.argmin(distances))
        return np.array(clusters)
    
    def update_centroids(self, data, clusters):
        new_centroids = []
        for i in range(self.k):
            cluster_points = data[clusters == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
        return np.array(new_centroids)
    
    def converged(self, old_centroids, new_centroids):
        return np.allclose(old_centroids, new_centroids)
    
    def run_kmeans(self, data, max_steps=100):
        self.initialize_centroids(data)
        for step in range(max_steps):
            clusters = self.assign_points_to_clusters(data)
            new_centroids = self.update_centroids(data, clusters)
            if self.converged(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return clusters
