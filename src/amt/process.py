import os
import argparse

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from preprocess.coco_format import *

def print_stats(coco):
    print("{} images".format(len(coco.dataset["images"])))
    print("{} annotations".format(len(coco.dataset["annotations"])))
    counts = {}
    for cat in coco.cats:
        catName = coco.cats[cat]["name"]
        annIds = coco.getAnnIds(catIds=[cat])
        counts[catName] = annIds

        print("{} {}".format(catName, len(annIds)))
    

def remove_duplicates(coco):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, default="../data/ade20k_val_predictions.json")
    parser.add_argument('-o', '--out_dir', type=str, default="../data/")
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    print_stats(coco)
