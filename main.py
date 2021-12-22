import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import os
from histogram_gen import *
from cosine_similarity import *
from calculate_d import *
from inferred_similarity import *
from split_input_video import *
from cooccurence_matrix import *
from quantize_histograms import *
from sklearn.manifold import TSNE


def main():
    files = []
    segment_count = 0
    for f in os.listdir("videos"):
        if f.endswith(".avi"):
            segment_count += 1
            files.append("videos/" + f)

    histograms = []
    for f in files:
        _, _, segment_thresh_array, N = frameDeltaGivenPureBackground(f, False)
        for segment in segment_thresh_array:
            histograms.append(segment)

    # arr: (? , (480*856) )
    # mappings: what indexs of arr map to what vid num
    arr, mappings = merge(histograms)

    K = 25
    labels, centers = quantize(arr, K)

    # segments needs to to be the N vdieo segments
    C_matrix = coMatrix(N, labels, mappings, K)
    C_transpose = np.transpose(C_matrix)
    identity = np.identity(N)

    beta = 1
    S_matrix = similarity(arr, K, labels)

    weight_matrix = np.zeros((N + K, N + K))
    weight_matrix[:N, :N] = identity
    weight_matrix[:N, N:] = C_transpose
    weight_matrix[N:, :N] = C_matrix
    weight_matrix[N:, N:] = beta * S_matrix

    D = calculate_d(weight_matrix)

    eigenvalues, eigenvectors = np.linalg.eig(D - weight_matrix)
    np.printoptions(threshold=np.inf)

    segmentToEigenCoordsDict = {}
    e_vects = []
    for segment in range(N):
        e_vects.append(eigenvectors[segment][:N])
        segmentToEigenCoordsDict[segment] = e_vects[segment]

    S = inferred_similarity(segmentToEigenCoordsDict, N)

    e_vects = np.array(e_vects)

    labels, centers = quantize(e_vects, 3)
    clusters = {}
    for i in range(len(labels)):
        try:
            entry = clusters[labels[i]]
        except:
            entry = []
        entry.append(i)
        clusters[labels[i]] = entry

    for i in range(len(labels)):
        if clusters[labels[i]] == 0:
            clusters[labels[i]] = []

    final = inter_cluster_similarity(clusters, S)
    print("final")
    print(final)

    tsne = TSNE(
        n_components=2, verbose=1, perplexity=5, n_iter=1000, learning_rate=200
    ).fit_transform(e_vects)
    normalized_tsne = (tsne - np.min(tsne)) / (np.max(tsne) - np.min(tsne))

    df_tsne = pd.DataFrame(normalized_tsne, columns=["comp1", "comp2"])
    df_tsne["label"] = np.arange(0, 7, 1)
    sns.lmplot(x="comp1", y="comp2", data=df_tsne, hue="label", fit_reg=False)
    plt.show()
    print("tsne", normalized_tsne)
    print("type", type(tsne))

    """
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in labels:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = np.array(labels[label], dtype=np.float) / 255

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc="best")

    # finally, show the plot
    plt.show()
    """

    return S


if __name__ == "__main__":
    S = main()
    np.set_printoptions(threshold=np.inf)
    print("S", S)
