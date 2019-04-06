import os
import argparse
import uuid
import random
import numpy as np
from tqdm import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from coco_utils.coco_format import save_ann_fn

def print_stats(coco):
    print("{} images, {} annotations".format(len(coco.dataset["images"]), len(coco.dataset["annotations"])))

def make_hidden_tests(cocoGt, cocoDt):
    # Adds detection segmentation to ground truth segmentation if IOU > 0.5
    matched = []
    for imgId in tqdm(cocoDt.imgs):
        gt_annIds = cocoGt.getAnnIds(imgIds=[imgId])
        dt_annIds = cocoDt.getAnnIds(imgIds=[imgId])
        gt_anns = cocoGt.loadAnns(gt_annIds)
        dt_anns = cocoDt.loadAnns(dt_annIds)
        gts = [ann["segmentation"] for ann in gt_anns]
        dts = [ann["segmentation"] for ann in dt_anns]
        iscrowds = [0 for _ in gts]
        if len(gts) == 0 or len(dts) == 0:
            continue

        ious = COCOmask.iou(dts, gts, iscrowds)
        max_ious = np.max(ious, axis=0)
        max_gt_ids = np.argmax(ious, axis=0)
        for gt_ann, max_iou, max_dt_id in zip(gt_anns, max_ious, max_gt_ids):
            dt_ann = dt_anns[max_dt_id]
            if max_iou > 0.5:
                hidden_test = {}
                hidden_test["segmentation"] = gt_ann["segmentation"]
                hidden_test["bbox"] = gt_ann["bbox"]
                hidden_test["area"] = gt_ann["area"]
                hidden_test["iou"] = max_iou

                gt_ann["segmentation"] = dt_ann["segmentation"]
                gt_ann["bbox"] = dt_ann["bbox"]
                gt_ann["area"] = dt_ann["area"]
                gt_ann["hidden_test"] = hidden_test
                matched.append(gt_ann)

    cocoGt.dataset["annotations"] = matched
    cocoGt.createIndex()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--ground_truth', type=str, default="../../LabelMe-Lite/data/ade20k/instances_val.json")
    parser.add_argument('-dt', '--detections', type=str, default="../../LabelMe-Lite/data/ade20k/maskrcnna_val.json")
    args = parser.parse_args()

    cocoGt = COCO(args.ground_truth)
    cocoDt = COCO(args.detections)
    print_stats(cocoGt)
    print_stats(cocoDt)

    make_hidden_tests(cocoGt, cocoDt)
    print_stats(cocoGt)

    out_file = "./test.json"
    images = cocoGt.dataset["images"]
    annotations = cocoGt.dataset["annotations"]
    categories = cocoGt.dataset["categories"]
    save_ann_fn(images, annotations, categories, out_file)
