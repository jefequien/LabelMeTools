import os
import json
import logging
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *

logger = logging.getLogger(__name__)

def print_stats(coco):
    print("{} images, {} annotations".format(len(coco.dataset["images"]), len(coco.dataset["annotations"])))

def correct_ann_size(coco):
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

def study(cocoGt, cocoDt):
    print_stats(cocoGt)
    print_stats(cocoDt)

    examples = []

    for imgId in tqdm(cocoDt.imgs):
        gt_annIds = cocoGt.getAnnIds(imgIds=[imgId])
        dt_annIds = cocoDt.getAnnIds(imgIds=[imgId])
        gt_anns = cocoGt.loadAnns(gt_annIds)
        dt_anns = cocoDt.loadAnns(dt_annIds)
        gts = [ann["segmentation"] for ann in gt_anns]
        dts = [ann["segmentation"] for ann in dt_anns]
        iscrowds = [0 for _ in dts]
        if len(gts) == 0 or len(dts) == 0:
            continue

        ious = COCOmask.iou(gts, dts, iscrowds)

        for i, gt_ious in enumerate(ious):
            gt_ann = gt_anns[i]
            max_iou = np.max(gt_ious)
            dt_idx = np.argmax(gt_ious)
            # print(gt_ann["id"], max_iou, dt_idx)
            if max_iou > 0.5:
                gt_ann["segmentation"] = dts[dt_idx]
                gt_ann["iou"] = max_iou
                examples.append(gt_ann)

    cocoGt.dataset["annotations"] = examples
    cocoGt.createIndex()
    print_stats(cocoGt)


if __name__ == "__main__":
    gt_fn = "../../LabelMe-Lite/data/ade20k/full_val.json"
    dt_fn = "../../LabelMe-Lite/data/ade20k/maskrcnnc_val.json"

    cocoGt = COCO(gt_fn)
    cocoDt = COCO(dt_fn)
    study(cocoGt, cocoDt)

    out_file = "./iou_examples.json"
    images = cocoGt.dataset["images"]
    annotations = cocoGt.dataset["annotations"]
    categories = cocoGt.dataset["categories"]
    save_ann_fn(images, annotations, categories, out_file)

