import os
import argparse
import numpy as np
import cv2

from dummy_datasets import get_ade_dataset
from coco_format import *

def make_annotations(ann_dir, im_list):
    annotations = []
    for imgId, im_name in enumerate(im_list):
        print(imgId, im_name, len(annotations))
        ann_path = os.path.join(ann_dir, im_name).replace('.jpg', '.png')

        ann_image = cv2.imread(ann_path)
        crowd_mask = ann_image[:,:,0]
        ins_mask = ann_image[:,:,1]
        cat_mask = ann_image[:,:,2]

        for ins in np.unique(ins_mask):
            if ins == 0:
                continue
            mask = (ins_mask == ins)
            cat = np.sum(cat_mask[mask]) / np.sum(mask)
            crowd = np.max(crowd_mask[mask])
            
            ann = make_ann(mask, cat, iscrowd=crowd)
            ann["image_id"] = imgId
            ann["id"] = len(annotations)
            annotations.append(ann)
    return annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="val")
    args = parser.parse_args()

    im_dir = "../data/ade20k/images/"
    ann_dir = "../data/ade20k/annotations/instances"
    cat_list = get_ade_dataset()
    im_list = []

    if args.split == "train":
        im_list = os.path.join(im_dir, "training.txt")
    elif args.split == "val":
        im_list = os.path.join(im_dir, "validation.txt")

    with open(im_list,'r') as f:
        im_list = f.read().splitlines()

    #images = make_images(im_dir, im_list)
    #categories = make_categories(cat_list)
    annotations = make_annotations(ann_dir, im_list)

    out_file = os.path.join(ann_dir, "instances_ade20k_{}.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_file)


