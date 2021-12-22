import numpy as np
from random import randint
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

#X is input matrix 
#n is number of clusters

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

'''

X = np.array([[1,1],
                [0,3],
                [4,4], 
                [6,0],
                [30,1],
                [31,3],
                [33,4], 
                [32,0],
                [10,30],
                [11,31],
                [12,32],
                [11,33],
                [40,41],
                [43,44],
                [40,40]])
centroids = furthest_point(X, 4, "okay")


kmeans = KMeans(n_clusters = 4, init = furthest_point)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
print(labels)
print(centers)
#print(X[:,1])
plt.scatter(X[:,1], X[:,0])
#print(centroids[:,1])
plt.scatter(centroids[:,1], centroids[:,0])
plt.show()
x0=[]
x1=[]
x2=[]
x3=[]
for i in range(len(labels)):
    if (labels[i] == 1):
        x1.append(X[i])
    if (labels[i]==0):
        x0.append(X[i])
    if (labels[i] == 2):
        x2.append(X[i])
    if (labels[i]==3):
        x3.append(X[i])
x0 = np.array(x0)
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
print(x0)
print(x1)
plt.scatter(x0[:,1], x0[:,0])
plt.scatter(x1[:,1], x1[:,0])
plt.scatter(x2[:,1], x2[:,0])
plt.scatter(x3[:,1], x3[:,0])
plt.show()
'''



