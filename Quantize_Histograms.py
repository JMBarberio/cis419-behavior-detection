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

from histogram_gen import frameDeltaGivenPureBackground


"""
  H_arr = [H1, H2, H3, ...]

  H_t = m x m

   
  fit predict(X) --> labels
    X: (n_samples, n_features)

  labels : (n_samples,  )
  cluster_centers: (n_clusters, n_features)
"""



# one prototype per frame aka one histogram per framke
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
  img_arr = frameDeltaGivenPureBackground("01_001.avi")
  print(img_arr)
  




