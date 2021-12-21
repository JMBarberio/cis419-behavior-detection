"""
Step 2 of Algorithm: 
quantize every histogram Ht into K prototypes: 
P = {p1, . . . , pK}


Steps
- apply vector quantization to the histogram feture vectors classifying them into a 
 dictionary of K prototype features, P = {p1,,,} USING K-means

output: 
1) the K protoypes aka centroids of the result of the k emeans algo
2) a mapping of all the histograms to what prototype they end up in
"""


import numpy as np
from sklearn import cluster

from histogram_gen import *


"""
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
    n = np.reshape(Ht, (-1, 1))
    X.append(n)


  temp_arr = np.asarray(X)
  #print(temp_arr.shape)
  temp_arr = np.reshape(temp_arr, (n_samples, -1))

  k_means = cluster.KMeans(n_clusters=K)
  labels = k_means.fit_predict(temp_arr)
  centers = k_means.cluster_centers_
  return labels, centers



if __name__ == "__main__":
  img_array, frame_delta_array, thresh_array = frameDeltaGivenPureBackground("01_001.avi", False)
  labels, centers = quantiize(thresh_array, 10)
  print(labels)
  print(centers)


  






