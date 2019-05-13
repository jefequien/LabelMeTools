import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

import sys
sys.path.append("../coco_utils")
from coco_format import *

def plot_bins(coco):
    iou_vs_accepted = []
    num_accept = np.zeros(11)
    num_total = np.zeros(11)
    for ann in coco.dataset["annotations"]:
        iou = ann["completed_task_sim"]["iou"]
        acc = ann["completed_task"]["accepted"]
        iou_vs_accepted.append([iou, acc])
        if acc:
            num_accept[round(iou*10)] += 1
        num_total[round(iou*10)] += 1

    p_accept = num_accept / num_total
    iou = np.linspace(0,1,11)

    title = "Probability of accepting"
    xlabel = "IOU"
    ylabel = "Prob of Accepted"
    x = iou
    y = p_accept
    plot_scatter(x, y, title, xlabel, ylabel)

def plot_p_accepted(coco):
    iou_vs_accepted = []
    for ann in coco.dataset["annotations"]:
        iou = ann["completed_task_sim"]["iou"]
        acc = ann["completed_task"]["accepted"]
        iou_vs_accepted.append([iou, acc])
    iou_vs_accepted.sort(key=lambda x: x[0], reverse=True)
    iou_vs_accepted = np.array(iou_vs_accepted, dtype=float)
    iou_vs_accepted[:,1] = np.cumsum(iou_vs_accepted[:,1]) / np.arange(1, iou_vs_accepted.shape[0] + 1)

    title = "Probability of accepting when IOU > x"
    xlabel = "IOU"
    ylabel = "Prob of Accepted"
    x = iou_vs_accepted[:,0]
    y = iou_vs_accepted[:,1]
    plot_scatter(x, y, title, xlabel, ylabel)

def plot_iou_vs_accepted(coco):
    iou_vs_accepted = []
    for ann in coco.dataset["annotations"]:
        iou = ann["completed_task_sim"]["iou"]
        acc = ann["completed_task"]["accepted"]
        iou_vs_accepted.append([iou, acc])
    iou_vs_accepted = np.array(iou_vs_accepted, dtype=float)

    title = "IOU vs Accepted"
    xlabel = "IOU"
    ylabel = "Accepted"
    x = iou_vs_accepted[:,0]
    y = iou_vs_accepted[:,1]
    plot_scatter(x, y, title, xlabel, ylabel)

def plot_scatter(x, y, title="Title", xlabel="x label", ylabel="y label"):
    colors = (0,0,0)
    area = np.pi*3
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def accuracy(coco):
    right = 0
    total = 0
    for ann in coco.dataset["annotations"]:
        acc = ann["completed_task"]["accepted"]
        acc_sim = ann["completed_task_sim"]["accepted"]
        if acc == acc_sim:
            right += 1
        total += 1
    print(right / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    accuracy(coco)
    plot_bins(coco)
    # plot_iou_vs_accepted(coco)
    # plot_p_accepted(coco)
