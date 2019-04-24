import os
import sys
sys.path.append("../coco_utils")
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from coco_format import *

def merge_cocos(coco0, coco1):
    images = []
    annotations = []
    categories = []
    filename_to_id = {}
    catname_to_id = {}

    for annId in coco0.anns:
        ann = coco0.anns[annId]
        img = coco0.imgs[ann["image_id"]]
        cat = coco0.cats[ann["category_id"]]

        if img["file_name"] not in filename_to_id:
            filename_to_id[img["file_name"]] = len(images) + 1
            images.append(img)
        img["id"] = filename_to_id[img["file_name"]]

        if cat["name"] not in catname_to_id:
            catname_to_id[cat["name"]] = len(categories) + 1
            categories.append(cat)
        cat["id"] = catname_to_id[cat["name"]]

        ann["id"] = len(annotations) + 1
        ann["image_id"] = img["id"]
        ann["category_id"] = cat["id"]
        annotations.append(ann)

    for annId in coco1.anns:
        ann = coco1.anns[annId]
        img = coco1.imgs[ann["image_id"]]
        cat = coco1.cats[ann["category_id"]]

        if img["file_name"] not in filename_to_id:
            filename_to_id[img["file_name"]] = len(filename_to_id) + 1
            images.append(img)
        img["id"] = filename_to_id[img["file_name"]]

        if cat["name"] not in catname_to_id:
            catname_to_id[cat["name"]] = len(catname_to_id) + 1
            categories.append(cat)
        cat["id"] = catname_to_id[cat["name"]]

        ann["id"] = len(annotations) + 1
        ann["image_id"] = img["id"]
        ann["category_id"] = cat["id"]
        annotations.append(ann)

    coco = COCO()
    coco.dataset["images"] = images
    coco.dataset["annotations"] = annotations
    coco.dataset["categories"] = categories
    coco.createIndex()
    print_ann_fn(coco)
    return coco


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, required=True)
    parser.add_argument('-o', '--out_fn', type=str)
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = os.path.join(args.in_dir, "merged.json")
    print(args)

    ann_fns = [os.path.join(args.in_dir, fn) for fn in os.listdir(args.in_dir) if ".json" in fn]

    coco = COCO()
    for ann_fn in tqdm(ann_fns):
        coco = merge_cocos(coco, COCO(ann_fn))

    save_ann_fn(coco.dataset["images"], coco.dataset["annotations"], coco.dataset["categories"], args.out_fn)
    print_ann_fn(args.out_fn)
