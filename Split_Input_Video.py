"""
modules Required:
-moviepy

Params-
T: the number (integer) of equal length segments for the video to be split into
fileName: the filepath as a string of the video you wish to segment. 

output: list of length T containing the T video segments stored as moviepy VideoFileClip objects.

"""
import moviepy.editor as mpy

def split_input_into_T_segments(T, fileName):
    video = mpy.VideoFileClip(fileName)
    video_duration = video.duration
    segment_length = video_duration/T
    start_time = 0
    end_time = segment_length
    output = []
    for i in range(T):
        #print(i)
        output.append(video.subclip(start_time, end_time))
        #print("segment created")
        start_time = end_time
        end_time += segment_length
    return output

'''
this was a test to make sure the output looks good
print("here1")
testVideos = split_input_into_T_segments(5, "testVideo.mp4")
print("here2")
print("output length =", len(testVideos))

for i in range(5):
    testVideos[i].write_videofile("outputTest%s.mp4" % i, audio = False)

'''



