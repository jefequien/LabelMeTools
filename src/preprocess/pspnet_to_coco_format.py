import os
import json
import numpy as np
import cv2

from pycocotools import mask as COCOmask
from dummy_datasets import *

def make_annotations(cm, pm):
    pm = pm / 255
    good = (pm > 0.5)
    anns = []
    for i in np.unique(cm):
        mask = (cm == i)
        mask[~good] = 0
        cat = i + 1

        mask = np.asfortranarray(mask)
        mask = mask.astype(np.uint8)
        segm = COCOmask.encode(mask)
        segm["counts"] = segm["counts"].decode('ascii')

        ann = {}
        ann["segmentation"] = segm
        ann["category_id"] = int(cat)
        anns.append(ann)
    return anns

def make_ann_fn(im_list, cat_list, im_dir, out_dir):
    images = []
    annotations = []
    categories = []

    cm_dir = os.path.join(out_dir, "cm")
    pm_dir = os.path.join(out_dir, "pm")

    # Categories
    for i, name in enumerate(cat_list):
        categories.append({"id": i, "name": name})

    for imgId, im_name in enumerate(im_list):
        print(imgId, im_name, len(annotations))
        im_path = os.path.join(im_dir, im_name)
        cm_path = os.path.join(cm_dir, im_name.replace('.jpg', '.png'))
        pm_path = os.path.join(pm_dir, im_name.replace('.jpg', '.png'))

        im = cv2.imread(im_path)
        cm = cv2.imread(cm_path, 0)
        pm = cv2.imread(pm_path, 0)
        print(np.unique(cm))
        if cm is None or pm is None:
            continue

        # Images
        img = {}
        img["file_name"] = im_name
        img["id"] = imgId
        img["height"] = im.shape[0]
        img["width"] = im.shape[1]
        images.append(img)

        # Annotations
        anns = make_annotations(cm, pm)
        for ann in anns:
            ann["image_id"] = imgId
            ann["id"] = len(annotations)
            annotations.append(ann)

    ann_fn = {}
    ann_fn["images"] = images
    ann_fn["annotations"] = annotations
    ann_fn["categories"] = categories
    return ann_fn

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default="ade_val")
    parser.add_argument('-o', '--out_dir', type=str, default="../data/ade20k/pspnet_predictions/")
    args = parser.parse_args()
    
    im_dir = None
    im_list = []
    cat_list = get_ade150_dataset()

    if "ade" in args.project:
        im_dir = "../data/ade20k/images/"
        if "train" in args.project:
            im_list = "../data/ade20k/images/training.txt"
        elif "val" in args.project:
            im_list = "../data/ade20k/images/validation.txt"

    elif "places":
        raise

    with open(im_list,'r') as f:
        im_list = f.read().splitlines()

    ann_fn = make_ann_fn(im_list, cat_list, im_dir, args.out_dir)
    out_file = os.path.join(args.out_dir, args.project + "_pspnet_predictions.json")

    with open(out_file, 'w') as f:
            json.dump(ann_fn, f, indent=2)





