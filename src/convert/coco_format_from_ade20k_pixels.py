import os
import argparse
import numpy as np
import cv2

from dummy_datasets import get_ade150_dataset
from coco_format import *

def make_annotations(ann_dir, im_list):
    ann_dir = os.path.join(ann_dir, "pixels")
    annotations = []
    for imgId, im_name in enumerate(im_list):
        print(imgId, im_name, len(annotations))
        ann_path = os.path.join(ann_dir, im_name).replace('.jpg', '.png')

        cat_mask = cv2.imread(ann_path, 0)
        for cat in np.unique(cat_mask):
            if cat == 0:
                continue
            mask = (cat_mask == cat)

            ann = make_ann(mask, cat)
            ann["image_id"] = imgId
            ann["id"] = len(annotations)
            annotations.append(ann)
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="val")
    args = parser.parse_args()

    im_dir = "../data/ade20k/images/"
    ann_dir = "../data/ade20k/annotations/"
    cat_list = get_ade150_dataset()
    im_list = []

    if args.split == "train":
        im_list = os.path.join(im_dir, "training.txt")
    elif args.split == "val":
        im_list = os.path.join(im_dir, "validation.txt")

    with open(im_list,'r') as f:
        im_list = f.read().splitlines()

    images = make_images(im_dir, im_list)
    categories = make_categories(cat_list)
    annotations = make_annotations(ann_dir, im_list)

    out_file = os.path.join(ann_dir, "pixels_ade20k_{}.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_file)


