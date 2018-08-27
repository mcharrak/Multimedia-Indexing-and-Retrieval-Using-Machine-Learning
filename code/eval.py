import xml.etree.ElementTree as ET
import numpy as np

"""
    Evaluate performances of segmentation
    - cuts: [(start_time, end_time), ...]
    - gt:   [(start_time, end_time), ...]
"""

def get_gt_from_xml(path_xml):
    tree = ET.parse(path_xml)
    turns = tree.getroot().findall('.//Turn')
    gt = np.zeros((len(turns),2))
    for i, turn in enumerate(turns):
        endTime = float(turn.get('endTime'))        # in seconds
        startTime = float(turn.get('startTime'))    # in seconds
        gt[i, 0] = startTime
        gt[i, 1] = endTime
    # ordering based on the first column (startTime)
    gt = gt[gt[:, 0].argsort()]
    return gt


def eval_performances(cuts, gt, eps):
    """
    Evaluate performances of a video segmentation algorithm
    :param cuts: list of real numbers of the seconds at which it was detected a cut
    :param gt: nx2 matrix where each row has a startTime and an endTime
    :param eps: real number for the window of allowance for the inclusion of the detected cut
    :return: recall, precision
    """
    startTimes = gt[:, 0]

    k = 0   # current starting index of gt

    gt_label = np.zeros(gt.shape[0])
    for i, cut in enumerate(cuts):
        for j in range(k,gt.shape[0]):
            st = gt[j,0]
            # hit
            if st - eps <= cut <= st + eps:
                gt_label[j] = 1
                k = j+1
                break

    # number of correctly detected cuts (hits)
    C = gt_label.sum() + np.finfo(float).eps
    # number of not detected cuts (miss)
    M = np.where(gt_label == 0)[0].sum() + np.finfo(float).eps
    # number of falsely detected cuts (false hits)
    F = len(cuts) - C + np.finfo(float).eps
    # print("\tHits:\t{}\n\tMiss:\t{}\n\tFalse hits:\t{}".format(C, M, F))
    # recall
    recall = float(C) / (C + M)
    # precision
    precision = float(C) / (C + F)

    return recall, precision


def auc(x, y):
    return np.trapz(y,x)


path_xml = "/home/luca/Scrivania/AthensProject/resources/annotations/06-11-22.trs"