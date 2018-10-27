import os
import json
import numpy as np
import cv2

from dummy_datasets import *

from pycocotools import mask as COCOmask

def ann_image_to_annotations(ann_image):
    ins_mask = ann_image[:,:,1]
    cat_mask = ann_image[:,:,2]
    anns = []
    for ins in np.unique(ins_mask):
        if ins == 0:
            continue
        mask = (ins_mask == ins)
        cat = np.sum(cat_mask[mask]) / np.sum(mask)
        mask = np.asfortranarray(mask)
        mask = mask.astype(np.uint8)
        segm = COCOmask.encode(mask)
        segm["counts"] = segm["counts"].decode('ascii')

        ann = {}
        ann["segmentation"] = segm
        ann["category_id"] = int(cat)
        ann["area"] = COCOmask.area(segm)
        anns.append(ann)
    return anns

def make_ann_fn(im_dir, ann_dir, im_list, cat_list):
    images = []
    annotations = []
    categories = []

    # Categories
    for i, name in enumerate(cat_list):
        categories.append({"id": i, "name": name})

    for imgId, im_name in enumerate(im_list):
        print(imgId, im_name, len(annotations))

        im_path = os.path.join(im_dir, im_name)
        ann_path = os.path.join(ann_dir, im).replace('.jpg', '.png')

        im = cv2.imread(im_path)
        ann_image = cv2.imread(ann_path)

        # Images
        img = {}
        img["file_name"] = im_name
        img["id"] = imgId
        img["height"] = im.shape[0]
        img["width"] = im.shape[1]
        images.append(img)
        print(img["id"], img["file_name"])

        # Annotations
        anns = ann_image_to_annotations(ann_image)
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
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('-m', '--mode', type=str, default="instances")
    args = parser.parse_args()

    im_dir = "./data/ade20k/images/"
    ann_dir = "./data/ade20k/annotations/"
    im_list = None
    cat_list = None
    out_file = os.path.join(ann_dir, "{}_ade20k_{}.json".format(args.mode, args.split))

    if args.split == "train":
        im_list = os.path.join(im_dir, "training.txt")
    elif args.split == "val":
        im_list = os.path.join(im_dir, "validation.txt")

    if args.mode == "instances":
        ann_dir = os.path.join(ann_dir, "instances")
        cat_list = get_ade_dataset()
    elif args.mode == "pixels":
        ann_dir = os.path.join(ann_dir, "pixels")

    with open(im_list,'r') as f:
        im_list = f.read().splitlines()

    ann_fn = make_ann_fn(im_dir, ann_dir, im_list, cat_list)

    with open(out_file, 'w') as f:
            json.dump(ann_fn, f, indent=2)

