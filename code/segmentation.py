import cv2
import os
import numpy as np
from tqdm import tqdm
import eval
import matplotlib.pyplot as plt


def scores_to_cuts(threshold, scores, all_files, frame_rate):
    cuts = np.where(scores > threshold)[0]     # indices of the locations in which there is a peak
    scene_cuts = idx_to_seconds(cuts=cuts, all_files=all_files, frame_rate=frame_rate)
    return scene_cuts


def idx_to_seconds(cuts, all_files, frame_rate):
    frames_involved = all_files[cuts + 1]
    secs = np.zeros(len(frames_involved))
    for i, frame_involved in enumerate(frames_involved):
        frame_n = get_frame_number_from_name(frame_involved)
        sec = float(frame_n)/frame_rate
        secs[i] = sec
    return secs


# given a frame file name, returns the integer number of the frame
def get_frame_number_from_name(frame_name):
    only_number_str = frame_name.split('.')[0]
    return int(only_number_str)


def sum_abs_diff(frames_path, all_files, threshold=None, frame_rate=25):
    """
    Sum of the absolute differences
    :param frames_path: path of the frames of the video
    :param all_files: list of all the files corresponding to the video frames
    :param threshold: the threshold to use to filter the peaks
    :param frame_rate: the frame rate of the video
    :return: nx2 numpy array where [i,0] = startTime_i and [i,1] = endTime_i
    """
    previous = None
    # DEBUGGING: to be removed
    all_files = all_files[:]
    all_files = np.array(all_files)
    scores = np.zeros(len(all_files) - 1)
    for i, file in tqdm(enumerate(all_files), total=len(all_files)):
        frame = cv2.imread(frames_path + file)
        #n_pixels = frame.size
        # print("{} {}".format(i, file))
        if previous is not None:
            score = cv2.absdiff(frame, previous).sum()
            #score /= n_pixels
            scores[i-1] = score
        previous = frame
    if not threshold:
        median = np.median(scores)
        print("The median difference is: {}".format(median))
        threshold = median*2

    # normalization
    max_s = scores.max()
    scores /= max_s

    # plotting scores
    # plt.plot(scores)
    # plt.plot(np.arange(len(scores)), np.repeat(threshold,len(scores)))
    # plt.xlabel("frames")
    # plt.ylabel("SAD")
    # plt.show()
    cuts = np.where(scores > threshold)[0]     # indices of the locations in which there is a peak
    scene_cuts = idx_to_seconds(cuts=cuts, all_files=all_files, frame_rate=frame_rate)
    return scene_cuts, scores


def histogram_differences(frames_path, all_files, threshold=None, frame_rate=25):
    """
    Sum of histogram differences:
    :param frames_path: look at sum_abs_diff
    :param all_files: look at sum_abs_diff
    :param threshold: look at sum_abs_diff
    :param frame_rate: look at sum_abs_diff
    :return: look at sum_abs_diff
    """
    previous = None
    # DEBUGGING: to be removed
    all_files = all_files[:]
    all_files = np.array(all_files)
    scores = np.zeros(len(all_files) - 1)
    for i, file in tqdm(enumerate(all_files), total=len(all_files)):
        frame = cv2.imread(frames_path + file)
        hist_0 = cv2.calcHist(images=[frame], channels=[0], mask=None, histSize=[256], ranges=[0, 256]).flatten()
        hist_1 = cv2.calcHist(images=[frame], channels=[1], mask=None, histSize=[256], ranges=[0, 256]).flatten()
        hist_2 = cv2.calcHist(images=[frame], channels=[2], mask=None, histSize=[256], ranges=[0, 256]).flatten()

        #append the three vectors
        hist = np.append(hist_0,np.append(hist_1, hist_2))
        # print("{} {}".format(i, file))
        if previous is not None:
            score = np.absolute(hist - previous).sum()
            scores[i-1] = score
        previous = hist
    if threshold is None:
        median = np.median(scores)
        print("The median difference is: {}".format(median))
        threshold = median*2

    # normalization
    max_s = scores.max()
    scores /= max_s
    #plotting scores
    plt.plot(scores)
    plt.plot(np.arange(len(scores)), np.repeat(threshold,len(scores)))
    plt.xlabel("frames")
    plt.ylabel("HD")
    plt.show()
    cuts = np.where(scores > threshold)[0]     # indices of the locations in which there is a peak
    scene_cuts = idx_to_seconds(cuts=cuts, all_files=all_files, frame_rate=frame_rate)
    return scene_cuts, scores


def segment_video(frames_path, method, threshold=None, frame_rate=25):
    """
        Video segmentation:

        Input:
            :param frames_path path of the frames of the video
            :param method the method used to segment
                - 'sum_abs_diff': sum of the absolute differences
        Output:
            :return list of tuples (start_shot, end_shot)
    """
    all_files = sorted(os.listdir(frames_path))
    if method == 'sum_abs_diff':
        return sum_abs_diff(frames_path, all_files, threshold, frame_rate)
    elif method == 'histogram_diff':
        return histogram_differences(frames_path, all_files, threshold, frame_rate)


frames_path = "/home/luca/Scrivania/AthensProject/resources/frames_video/"
frame_rate = 25 #frames/sec
threshold = 0.15

segments, scores = segment_video(frames_path=frames_path, method='sum_abs_diff',
                         frame_rate=frame_rate, threshold=threshold)

# evalutation
gt = eval.get_gt_from_xml(eval.path_xml)

eps = 0.2

print("Precision-Recall curve")
thresholds = np.arange(0.01, 0.99, 0.01)
all_files = np.array(sorted(os.listdir(frames_path)))

recall_tot = np.zeros(len(thresholds))
precision_tot = np.zeros(len(thresholds))

for i, t in enumerate(thresholds):
    cuts = scores_to_cuts(threshold=t, scores=scores, all_files=all_files, frame_rate=frame_rate)
    recall, precision = eval.eval_performances(cuts=cuts, gt=gt, eps=eps)
    recall_tot[i] = recall
    precision_tot[i] = precision

plt.plot(recall_tot, precision_tot)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

idx = recall_tot.argsort()
recall_tot = recall_tot[idx]
precision_tot = precision_tot[idx]

auc = eval.auc(x=recall_tot, y=precision_tot)

print("AUC:\t{}".format(auc))


