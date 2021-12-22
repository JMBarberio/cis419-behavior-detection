# step 1: get motion histograms for each frame t

# step 2: quantize into prototypes

# step 3: slice video into T second long segments

# step 4: co-occurange matrix

# step 5: similarity step

# step 6: construct G with meight matrix W

# step 7: sovle for smallest eigen fectors

# step 8: compute ingerred similarity


import os
from histogram_gen import *
from quantize_histograms import *
from cosine_similarity import *
from cooccurence_matrix import *
from split_input_video import *

if __name__ == "__main__":
    files = []
    for f in os.listdir("videos"):
        files.append("videos/" + f)

    histograms = []
    for f in files:
        _, _, thresh_array = frameDeltaGivenPureBackground(f, False)
        histograms.append(mash(thresh_array))

    # arr: (? , (480*856) )
    # mappings: what indexs of arr map to what vid num
    arr, mappings = merge(histograms)

    labels, centers = quantize(arr, 25)

    # segments needs to to be the N vdieo segments
    segments = "beep"
    C_matrix = coMatrix(segments, labels, mappings)

    S_matrix = cosine_similarity(thresh_array)

    # testVideos = split_input_into_T_segments(5, filename)
    # print("here2")
    # print("output length =", len(testVideos))
