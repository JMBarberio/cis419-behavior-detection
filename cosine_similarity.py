from cv2 import THRESH_TOZERO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cs
from histogram_gen import *


def cosine_similarity(thresh_array):
    gamma = 1.0
    K = len(thresh_array)
    chi2_list = []
    avg_chi2_list = []
    S_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            zero = thresh_array[i].reshape(1, -1)
            one = thresh_array[j].reshape(1, -1)
            kernel = cs(zero, one)
            chi2_list.append(kernel)
            avg = np.sum(kernel) / np.size(kernel)
            avg_chi2_list.append(avg)
            S_matrix[i, j] = avg

    np.set_printoptions(threshold=np.inf)
    print("avg", avg)
    print("kernel", kernel)
    print("S", S_matrix)

    return S_matrix