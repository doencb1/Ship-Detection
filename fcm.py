import numpy as np

class FuzzyCMeans:
    def __init__(self, n_clusters=2, m=2, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.U = None

    def initialize_centers(self, X):
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centers = X[random_indices]
        return centers

    def initialize_centers_kmeans_plusplus(self, X):
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features))
        
        centers[0] = X[np.random.choice(n_samples)]
        
        for c_id in range(1, self.n_clusters):
            distances = np.min([np.sum((X - center) ** 2, axis=1) for center in centers[:c_id]], axis=0)
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            
            next_center = X[np.searchsorted(cumulative_probabilities, r)]
            centers[c_id] = next_center
        
        return centers

    def initialize_membership_matrix(self, n_samples):
        U = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)
        return U

    def update_centers(self, X):
        um = self.U ** self.m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        return centers

    def update_membership_matrix(self, X):
        n_samples = X.shape[0]
        new_U = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                denominator = sum((np.linalg.norm(X[i] - self.centers[j]) / np.linalg.norm(X[i] - self.centers[k])) ** (2 / (self.m - 1)) for k in range(self.n_clusters))
                new_U[i, j] = 1 / denominator          
        return new_U
    
    def objective_function(self, X):
        um = self.U ** self.m
        obj = np.sum(um * np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2) ** 2)
        return obj
    
    def fit(self, X):
        n_samples = X.shape[0]
        self.U = self.initialize_membership_matrix(n_samples)
        
        self.centers = self.initialize_centers_kmeans_plusplus(X)

        for iteration in range(self.max_iter):
            self.centers = self.update_centers(X)
            new_U = self.update_membership_matrix(X)
            
            objective_value = self.objective_function(X)
            print(f'FCM: {iteration + 1}, Objective function: {objective_value}')

            if np.linalg.norm(new_U - self.U) < self.tol:
                print(f'Converged after {iteration + 1} iterations')
                break
            
            self.U = new_U
            
    def predict(self, X):
        new_U = self.update_membership_matrix(X)
        return np.argmax(new_U, axis=1)
