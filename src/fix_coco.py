import os
import json
import logging
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *

def fix_coco(coco):
    for imgId in tqdm(coco.imgs):
        img = coco.imgs[imgId]
        h = img["height"]
        w = img["width"]

        annIds = coco.getAnnIds(imgIds=[imgId])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            segm = ann["segmentation"]
            if segm["size"] != [h,w]:
                mask = COCOmask.decode(segm)
                new_mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
                new_ann = make_ann(new_mask)

                ann["segmentation"] = new_ann["segmentation"]
                ann["area"] = new_ann["area"]
                ann["bbox"] = new_ann["bbox"]



if __name__ == "__main__":
    ann_fn = "../../LabelMe-Lite/data/ade20k/full_val.json"
    coco = COCO(ann_fn)

    fix_coco(coco)

    out_file = "./iou_examples.json"
    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, out_file)