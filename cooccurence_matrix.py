"""

C : K by N 
C(i, j) = 1 iff segment vj contains an
image frame whose image feature is classified as prototype pi


Working under the assumptions we have the following information handy:
- Need to know what histograms correspond to what video segment
- labels: index of the cluster each sampe belongs to
    size (N_samples, )
- centers= the actual K different protoypes


Inputs
- V_arr: N video segments 

Output: C 
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import cluster


def prototypesFromFrames(frames, labels):
    ps = []
    for i in frames:
        ps.append(labels[i])
    return ps


# mappings: dict. what histograms correspond to what video segment
def coMatrix(segments, labels, mappings, K):
    N = segments  # video segments
    cMatrix = np.zeros((K, N))

    for j in range(0, N):
        frames = mappings[j]
        ps = prototypesFromFrames(frames, labels)
        for i in range(0, K):
            if i in ps:
                cMatrix[i, j] = 1

    return cMatrix
