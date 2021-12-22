from cv2 import THRESH_TOZERO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import chi2_kernel, cosine_similarity


def similarity(thresh_array, K, labels):
    dictionary = {}
    for i in range(len(labels)):
        dictionary[labels[i]] = i

    gamma = 1.0
    chi2_list = []
    avg_chi2_list = []
    S_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            try:
                zero = thresh_array[dictionary[i]].reshape(1, -1)
                one = thresh_array[dictionary[j]].reshape(1, -1)
                kernel = chi2_kernel(zero, one)
                chi2_list.append(kernel)
                avg = np.sum(kernel) / np.size(kernel)
                avg_chi2_list.append(avg)
                S_matrix[i, j] = avg
            except KeyError as e:
                S_matrix[i, j] = 0
                print("No histograms associated with i: ", i, " j: ", j)

    return S_matrix
