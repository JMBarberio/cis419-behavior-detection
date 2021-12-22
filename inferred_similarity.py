import numpy as np


def inferred_similarity(segmentToEigenCoordsDict, numberOfSegments):
    S = np.zeros((numberOfSegments, numberOfSegments))
    for i in range(numberOfSegments):
        for j in range(i, numberOfSegments):
            vi = segmentToEigenCoordsDict[i]
            vj = segmentToEigenCoordsDict[j]
            if i == 1:
                print(vi - vj)
            S[i][j] = np.exp(-np.linalg.norm(vi - vj, ord=1))
            if i != j:
                S[j][i] = S[i][j]
    return S


# v1 = np.array([3,4,2])
# v2 = np.array([7,1,2])
# v3 = np.array([0,0,0])
# v4 = np.array([1,2,4])
# v5 = np.array([0,0,0])
# v6 = np.array([7,4,1])

# dict = {0:v1, 1:v2, 2:v3, 3:v4, 4:v5, 5:v6}
# print(inferred_similarity(dict, 6))


def inter_cluster_similarity(clusterList, S, N):
    SVC = np.zeros((len(clusterList), len(clusterList)))
    for i in range(len(clusterList)):
        for j in range(i, len(clusterList)):
            for k in range(len(clusterList[i])):
                for l in range(len(clusterList[j])):
                    SVC[i][j] += S[clusterList[i][k]][clusterList[j][l]]
    return SVC
