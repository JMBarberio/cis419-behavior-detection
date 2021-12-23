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


def inter_cluster_similarity(clusterList, S):
    SVC = np.zeros((len(clusterList), len(clusterList)))
    for i in range(len(clusterList)):
        for j in range(i, len(clusterList)):
            for k in range(len(clusterList[i])):
                for l in range(len(clusterList[j])):
                    SVC[i][j] += S[clusterList[i][k]][clusterList[j][l]]
                    SVC[j][i] = SVC[i][j]
    return SVC
