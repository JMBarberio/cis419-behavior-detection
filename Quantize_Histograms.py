"""
Step 2 of Algorithm: 
quantize every histogram Ht into K prototypes: 
P = {p1, . . . , pK}

modules Required:


Steps
- apply vector quantization to the histogram feture vectors classifying them into a 
 dictionary of K prototype features, P = {p1,,,} USING K-means

 Inputs
 - Histogram H
 - K 

output: 
1) the K protoypes aka centroids of the result of the k emeans algo
2) a mapping of all the histograms to what prototype they end up in
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.misc import face


"""
  H_arr = [H1, H2, H3, ...]

  H_t = m x m

   
  fit predict(X) --> labels
    X: (n_samples, n_features)

  labels : (n_samples,  )
  cluster_centers: (n_clusters, n_features)
"""

def quantiize(H_arr, K):
  n_samples = len(H_arr)
  X = []
  for Ht in H_arr:
    n = Ht.reshape((-1, 1))
    X.append(n)

  print(X)
  print(X.shape)
  
  X = X.reshape((-1, 1))  # We need an (n_sample, n_feature) array
  k_means = cluster.KMeans(n_clusters=K)
  labels = k_means.fit_predict(H_arr)
  centers = k_means.cluster_centers_
  return labels, centers


if __name__ == "__main__":
  n_clusters = 5
  np.random.seed(0)
  face = face(gray=True)

  print(face.shape)

  X = face.reshape((-1, 1))  # We need an (n_sample, n_feature) array
  print(X.shape)
  k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
  k_means.fit(X)
  values = k_means.cluster_centers_.squeeze()
  labels = k_means.labels_

  # create an array from labels and values
  face_compressed = np.choose(labels, values)
  face_compressed.shape = face.shape

  vmin = face.min()
  vmax = face.max()

  # original face
  plt.figure(1, figsize=(3, 2.2))
  plt.imshow(face, cmap=plt.cm.gray, vmin=vmin, vmax=256)

  # compressed face
  plt.figure(2, figsize=(3, 2.2))
  plt.imshow(face_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

  # equal bins face
  regular_values = np.linspace(0, 256, n_clusters + 1)
  regular_labels = np.searchsorted(regular_values, face) - 1
  regular_values = 0.5 * (regular_values[1:] + regular_values[:-1])  # mean
  regular_face = np.choose(regular_labels.ravel(), regular_values, mode="clip")
  regular_face.shape = face.shape
  plt.figure(3, figsize=(3, 2.2))
  plt.imshow(regular_face, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

  # histogram
  plt.figure(4, figsize=(3, 2.2))
  plt.clf()
  plt.axes([0.01, 0.01, 0.98, 0.98])
  plt.hist(X, bins=256, color=".5", edgecolor=".5")
  plt.yticks(())
  plt.xticks(regular_values)
  values = np.sort(values)
  for center_1, center_2 in zip(values[:-1], values[1:]):
      plt.axvline(0.5 * (center_1 + center_2), color="b")

  for center_1, center_2 in zip(regular_values[:-1], regular_values[1:]):
      plt.axvline(0.5 * (center_1 + center_2), color="b", linestyle="--")

  plt.show()




