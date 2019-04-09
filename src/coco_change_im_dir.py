import os
import argparse
import json
import logging
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *

def set_new_im_dir(coco, old_im_dir, new_im_dir):
    for imgId in tqdm(coco.imgs):
        img = coco.imgs[imgId]
        im_name = img["file_name"]
        full_path = os.path.join(old_im_dir, im_name)
        img["file_name"] = os.path.relpath(full_path, new_im_dir)


def resize_annotations(coco):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')

    parser.add_argument('-n', '--new_im_dir', type=str, help='New image directory')
    parser.add_argument('-d', '--old_im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    args = parser.parse_args()
    print(args)

    coco = COCO(args.ann_fn)
    out_fn = args.ann_fn.replace(".json", "_fixed.json")

    change_im_dir(coco, args.old_im_dir, args.new_im_dir)
    # resize_annotations(coco)

    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, args.out_fn)
