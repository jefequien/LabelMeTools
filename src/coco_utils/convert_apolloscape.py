import os
import argparse
import numpy as np
import cv2

from coco_format import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="val")
    args = parser.parse_args()

    data_dir = "../data/ade20k"
    im_dir = os.path.join(data_dir, "images")
    ann_dir = os.path.join(data_dir, "annotations")

    