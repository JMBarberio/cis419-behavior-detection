"""
Step 2 of Algorithm: 
quantize every histogram Ht into K prototypes: 
P = {p1, . . . , pK}

output: 
1) the K protoypes aka centroids of the result of the k emeans algo
2) a mapping of all the histograms to what prototype they end up in

  fit predict(X) --> labels
    X: (n_samples, n_features)

  labels : (n_samples,  )
  cluster_centers: (n_clusters, n_features)
"""


import numpy as np
from sklearn import cluster
from histogram_gen import *


# take a thresh and reshape it so its 2d and not 2d
def mash(thresh):
    n_samples = len(thresh)
    X = []
    for Ht in thresh:
        n = np.reshape(Ht, (-1, 1))
        X.append(n)
    temp_arr = np.asarray(X)
    temp_arr = np.reshape(temp_arr, (n_samples, -1))
    return temp_arr


# merge all histograms into one array
def merge(arr):
    merged = arr[0]
    mappings = {}  # keep track of what frame is to what video?

    count = 0  # what actual hist # it is
    index = 0  # index to start at in mapping
    clip_num = 0

    for i in arr:
        l = len(i)
        mappings[clip_num] = [i + index for i in range(0, l)]
        index = index + l
        clip_num += 1

        if count == 0:
            count += 1
            continue
        else:
            count += 1
            x = np.vstack((merged, i))
            merged = x

    return merged, mappings


def quantize(hists, K):
    k_means = cluster.KMeans(n_clusters=K)
    labels = k_means.fit_predict(hists)
    centers = k_means.cluster_centers_
    return labels, centers
