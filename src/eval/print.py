import os
import json
import logging
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

def print_stats(coco, threshold=None):
    print("{} images, {} annotations".format(len(coco.dataset["images"]), len(coco.dataset["annotations"])))

    for cat in coco.cats:
        catName = coco.cats[cat]["name"]

        annIds = coco.getAnnIds(catIds=[cat])
        count = len(annIds)
        if threshold != None:
            anns = coco.loadAnns(ids=annIds)
            count = len([a for a in anns if a["score"] > threshold])
        print(catName, count)


if __name__ == "__main__":
    # ann_fn = "../data/000_maskrcnna.json"
    # ann_fn = "../data/ade20k/annotations/instances_ade20k_val.json"
    
    # ann_fn = "../data/ade20k/predictions/maskrcnn_ade/amt/predictions.json"
    ann_fn = "../data/ade20k/predictions/maskrcnn_coco/amt/predictions.json"
    coco = COCO(ann_fn)
    
    # print_stats(coco)
    print_stats(coco, threshold=0.5)