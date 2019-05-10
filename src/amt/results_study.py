import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

import sys
sys.path.append("../coco_utils")
from coco_format import *

def study(coco):
    iou_acc = []
    num_accept = np.zeros(11)
    num_total = np.zeros(11)
    for ann in coco.dataset["annotations"]:
        iou = ann["completed_task_sim"]["iou"]
        acc = ann["completed_task"]["accepted"]
        iou_acc.append([iou, acc])
        if acc:
            num_accept[round(iou*10)] += 1
        num_total[round(iou*10)] += 1

    p_accept = num_accept / num_total
    iou = np.linspace(0,1,11)
    # plot_scatter(iou, p_accept)

    # iou_acc.sort(key=lambda x: x[0], reverse=True)
    # iou_acc[:,1] = np.cumsum(iou_acc[:,1]) / np.arange(1, iou_acc.shape[0] + 1)
    iou_acc = np.array(iou_acc, dtype=float)
    plot_scatter(iou_acc[:,0], iou_acc[:,1])

def plot_scatter(x, y):
    colors = (0,0,0)
    area = np.pi*3
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title('Worker')
    plt.xlabel('IOU')
    plt.ylabel('Probability of accept')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    study(coco)
