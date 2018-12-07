import os
import argparse
import uuid
import random
import json
import numpy as np

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from preprocess.coco_format import save_ann_fn

# Bundles are Coco annotation files

def make_bundle_for_category(coco, catId, out_file):
    images = []
    annotations = []
    categories = [coco.cats[catId]]
    imgIds = set()

    for annId in coco.anns:
        ann = coco.anns[annId]
        if ann["category_id"] == catId:
            annotations.append(ann)
            imgIds.add(ann["image_id"])

    for imgId in imgIds:
        images.append(coco.imgs[imgId])

    print("{}: {} annotations, {} images".format(coco.cats[catId]["name"], len(annotations), len(images)))
    save_ann_fn(images, annotations, categories, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, default="../data/test/detections_0.5_#.json")
    parser.add_argument('-o', '--out_dir', type=str, default="../data/test/")
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    for catId in coco.cats:
        if (catId == 0):
            continue

        out_dir = os.path.join(args.out_dir, os.path.basename(args.ann_fn).replace(".json", ""))
        out_fn = os.path.join(out_dir, "{}_#.json".format(catId))
        bundle = make_bundle_for_category(coco, catId, out_fn)


