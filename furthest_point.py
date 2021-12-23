import numpy as np
from random import *
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#X is input matrix 
#n is number of clusters

seed(999)

def furthest_point(X, n_clusters, random_state):
    centroids = []
    
    startindex = randint(0, len(X) - 1)
    start = X[startindex]
    centroids.append(start)
    distances = np.zeros(len(X))
    distances[startindex] = -np.inf
    for c in range(n_clusters-1):
        for i in range(len(X)):
            distances[i] += np.linalg.norm(X[startindex] - X[i])
        startindex = np.argmax(distances)
        centroids.append(X[startindex])
        distances[startindex] = -np.inf
    return np.array(centroids)



