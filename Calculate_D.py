import numpy as np


def calculate_D(W):
    shape = W.shape
    if shape[0] != shape[1]:
        print("W must be a square matrix")
        return -1
    D = np.zeros(shape)
    sum = np.sum(W, axis=0)
    for i in range(shape[0]):
        D[i][i] = sum[i]
    return D


# input = np.array([[3,4,1],[5,6,10],[8,3,9]])
# output = calculate_D(input)
# print(input)
# print(output)
