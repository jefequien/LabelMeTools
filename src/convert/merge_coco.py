import argparse
import os
import cv2
import numpy as np

from pycocotools.coco import COCO
from coco_format import *

def remove_duplicate_images(coco):
    name_to_id = {}
    images = []
    for imgId in coco.imgs:
        img = coco.imgs[imgId]
        name = img["file_name"]
        if name in name_to_id:
            newId = name_to_id[name]
            annIds = coco.getAnnIds(imgIds=[imgId])
            for annId in annIds:
                ann = coco.anns[annId]
                ann["image_id"] = newId
        else:
            name_to_id[name] = imgId
            images.append(img)

    coco.dataset["images"] = images
    return coco

def remove_duplicate_categories(coco):
    name_to_id = {}
    categories = []
    for catId in coco.cats:
        cat = coco.cats[catId]
        name = cat["name"]
        if name in name_to_id:
            newId = name_to_id[name]
            annIds = coco.getAnnIds(catIds=[catId])
            for annId in annIds:
                ann = coco.anns[annId]
                ann["category_id"] = newId
        else:
            name_to_id[name] = catId
            categories.append(cat)

    coco.dataset["categories"] = categories
    return coco

def merge_cocos(coco0, coco1):
    # Merge images
    for imgId in coco1.imgs:
        img = coco1.imgs[imgId]
        newId = len(coco0.dataset["images"])
        annIds = coco1.getAnnIds(imgIds=[imgId])
        for annId in annIds:
            ann = coco1.anns[annId]
            ann["image_id"] = newId

        img["id"] = newId
        coco0.dataset["images"].append(img)

    # Merge categories
    for catId in coco1.cats:
        cat = coco1.cats[catId]
        newId = len(coco0.dataset["categories"])
        annIds = coco1.getAnnIds(catIds=[imgId])
        for annId in annIds:
            ann = coco1.anns[annId]
            ann["category_id"] = newId

        cat["id"] = newId
        coco0.dataset["categories"].append(cat)

    # Merge annotations
    for annId in coco1.anns:
        ann = coco1.anns[annId]
        newId = len(coco0.dataset["annotations"])
        ann["id"] = newId
        coco0.dataset["annotations"].append(ann)
    coco0.createIndex()
    return coco0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, required=True)
    args = parser.parse_args()

    out_fn = os.path.join(args.outdir, "merged.json")
    ann_fns = [os.path.join(args.indir, fn) for fn in os.listdir(args.indir) if ".json" in fn]

    coco = COCO()
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco.dataset["categories"] = []

    for ann_fn in ann_fns:
        coco = merge_cocos(coco, COCO(ann_fn))
    coco = remove_duplicate_images(coco)
    coco = remove_duplicate_categories(coco)

    save_ann_fn(coco.dataset["images"], coco.dataset["annotations"], coco.dataset["categories"], out_fn)
    open_coco(out_fn)
