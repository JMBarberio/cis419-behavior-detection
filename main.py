import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2
import os
from numba import jit
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
    segment_frame_mapping = []
    N_counter = 0
    for f in files:
        _, _, segment_thresh_array, N, segment_frame_list, img_size = frameDeltaGivenPureBackground(f, False)
        N_counter += N
        segment_frame_mapping.extend(segment_frame_list)
        for segment in segment_thresh_array:
            histograms.append(segment)
    
    print(np.size(segment_frame_mapping))
    N = N_counter

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

    _, eigenvectors = np.linalg.eig(D - weight_matrix)

    segmentToEigenCoordsDict = {}
    e_vects = []
    for segment in range(N):
        e_vects.append(eigenvectors[segment][:N])
        segmentToEigenCoordsDict[segment] = e_vects[segment]

    S = inferred_similarity(segmentToEigenCoordsDict, N)

    e_vects = np.array(e_vects)

    labels, centers = quantize(e_vects, 10)

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
    print("final", final)

    finalnums = np.zeros(len(final))

    for i in range(len(final)):
        for j in range(len(final[i])):
            finalnums[i]+= final[i][j]
    sum = np.sum(finalnums)
    for i in range(len(finalnums)):
        finalnums[i] = finalnums[i]/sum
    finalbools = np.zeros(len(finalnums))
    thresholdval = 0.015
    for i in range(len(finalbools)):
        if (finalnums[i] < thresholdval):
            finalbools[i] = 1
    print('finalbools', finalbools)
    print('finalnums', finalnums)

    for i in range(len(finalnums)):
        print("cluster", i, "has", len(clusters[i]),"segments")
        if finalnums[i] < thresholdval:
            print("cluster: ", clusters[i])
            for index in clusters[i]:
                name = 'segment'+str(index)+'.png'
                cv2.imwrite('segment1'+str(index)+'.png', segment_frame_mapping[index][0])


                        
 
    

    tsne = TSNE(
        n_components=2, verbose=1, perplexity=5, n_iter=1000, learning_rate=200
    ).fit_transform(e_vects)
    normalized_tsne = (tsne - np.min(tsne)) / (np.max(tsne) - np.min(tsne))

    df_tsne = pd.DataFrame(normalized_tsne, columns=["comp1", "comp2"])
    #df_tsne["label"] = np.arange(0, N, 1)
    listofx=[]
    listofy=[]
    for i in range(len(clusters)):
        listx = []
        listy = []
        for j in range(len(clusters[i])):
            listx.append(df_tsne['comp1'][clusters[i][j]])
            listy.append(df_tsne['comp2'][clusters[i][j]])
        listofx.append(listx)
        listofy.append(listy)

    sns.lmplot(x="comp1", y="comp2", data=df_tsne, fit_reg=False)
    plt.show()
    for i in range(len(listofx)):
        plt.scatter(listofx[i], listofy[i])
    plt.show()



    
    return S


if __name__ == "__main__":
    S = main()
    np.set_printoptions(threshold=np.inf)
