import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import sys

# Serial program for data validation. This will be the oracle that tells if the data produced by our program is any good
# Note that the initialization of the data is random instead of being first k elements. 
# This can be a good indicator of how much different the result is with different initial centroid
# and if we converge **close** enough it means our approach is just as correct.

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2,axis=1))
    
    def fit(self, X, max_iter=1):
        self.centroids = np.random.uniform(np.amin(X,axis=0),np.amax(X,axis=0),size=(self.k,X.shape[1]))

        for _ in range(max_iter):
            y = []
            for data_points in X:
                distances = KMeansClustering.euclidean_distance(data_points, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
            y = np.array(y)

            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices],axis=0)[0])
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return y

# random_points = np.random.randint(0,100,(100,2))
file_path = "..//data_original.csv"
try:
    with open(file_path, 'r') as f:
        dataset = read_csv(file_path)
        pass
except FileNotFoundError:  # This is skipped if file exists
    print("FileNotFoundError")
    sys.exit()
finally:
    pass


# Initialize the dataset used for the computing
dataset = np.column_stack((dataset["x"],dataset["y"]))

kmeans = KMeansClustering(int(sys.argv[1]))
labels = kmeans.fit(dataset)

# Show the resulting data
plt.scatter(dataset[:, 0], dataset[:,1],c=labels,s=1)
plt.scatter(kmeans.centroids[: ,0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),marker="*",s=150)
plt.show()