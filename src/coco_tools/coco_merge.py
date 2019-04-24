import os
import sys
sys.path.append("../coco_utils")
import argparse
import cv2
import numpy as np

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
            filename_to_id[img["file_name"]] = len(filename_to_id) + 1
            img["id"] = filename_to_id[img["file_name"]]
            images.append(img)
        if cat["name"] not in catname_to_id:
            catname_to_id[cat["name"]] = len(catname_to_id) + 1
            cat["id"] = catname_to_id[cat["name"]]
            categories.append(cat)
        ann["id"] = len(annotations) + 1
        annotations.append(ann)

    for annId in coco1.anns:
        ann = coco1.anns[annId]
        img = coco1.imgs[ann["image_id"]]
        cat = coco1.cats[ann["category_id"]]

        if img["file_name"] not in filename_to_id:
            filename_to_id[img["file_name"]] = len(filename_to_id) + 1
            img["id"] = filename_to_id[img["file_name"]]
            images.append(img)
        if cat["name"] not in catname_to_id:
            catname_to_id[cat["name"]] = len(catname_to_id) + 1
            cat["id"] = catname_to_id[cat["name"]]
            categories.append(cat)
        ann["id"] = len(annotations) + 1
        annotations.append(ann)

    coco = COCO()
    coco.dataset["images"] = images
    coco.dataset["annotations"] = annotations
    coco.dataset["categories"] = categories
    coco.createIndex()
    return coco


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, required=True)
    args = parser.parse_args()

    out_fn = os.path.join(args.outdir, "merged.json")
    ann_fns = [os.path.join(args.indir, fn) for fn in os.listdir(args.indir) if ".json" in fn]

    coco = COCO()
    for ann_fn in ann_fns:
        coco = merge_cocos(coco, COCO(ann_fn))

    save_ann_fn(coco.dataset["images"], coco.dataset["annotations"], coco.dataset["categories"], out_fn)
    print_ann_fn(out_fn)
