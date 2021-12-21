"""
Step 4 of Algorithm: 

compute the co-occurrence matrix C(i, j) ∈ RK×N
between each prototype feature pi and video segment vj
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


def get_frames(t):
  return 


def prototypesFromFrames(t):
  return



def getCoMatrix(V_arr, labels):
  K = 100
  N = len(V_arr) # video segments
  cMatrix = np.zeros((K, N))

  for j in range(0,N):
    frames = get_frames(j)
    ps = prototypesFromFrames(frames)
    for i in range(0,K):
      if i in ps:
        cMatrix[i,j] = 1

  return cMatrix





if __name__ == "__main__":
    data = "testing"




