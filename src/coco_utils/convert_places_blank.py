import os
import argparse
import numpy as np
import glob
import cv2

from dummy_datasets import get_ade_dataset
from coco_format import *


def make_im_list(im_dir):
    im_list = []
    for filename, folder in zip(filenames, folders):
        im_path = os.path.join(folder, filename)
        if "ADE_" + split in im_path:
            im_list.append(im_path)
    return im_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--im_dir', required=True)
    args = parser.parse_args()

    # Load im_list
    im_list = make_im_list(args.im_dir)

    # Load cat_list
    cat_list = get_ade_dataset()

    annotations = []
    images = make_images(im_list, args.im_dir)
    categories = make_categories(cat_list)

    out_file = "places_ade100.json"
    save_ann_fn(images, annotations, categories, out_file)