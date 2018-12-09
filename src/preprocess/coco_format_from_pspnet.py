import os
import argparse
import numpy as np
import cv2

from dummy_datasets import get_ade150_dataset
from coco_format import *

def make_annotations(cm, pm):
    anns = []
    for i in np.unique(cm):
        mask = (cm == i)
        mask[~good] = 0
        cat = i + 1

        mask = np.asfortranarray(mask)
        mask = mask.astype(np.uint8)
        segm = COCOmask.encode(mask)
        segm["counts"] = segm["counts"].decode('ascii')

        ann = {}
        ann["segmentation"] = segm
        ann["category_id"] = int(cat)
        ann["area"] = int(COCOmask.area(segm))
        ann["bbox"] = list(COCOmask.toBbox(segm))
        ann["iscrowd"] = 0
        anns.append(ann)
    return anns

def make_annotations(ann_dir, im_list):
    print("Making annotations...")
    annotations = []

    cm_dir = os.path.join(ann_dir, "cm")
    pm_dir = os.path.join(ann_dir, "pm")

    for imgId, im_name in enumerate(im_list):
        print(imgId, im_name, len(annotations))
        cm_path = os.path.join(cm_dir, im_name.replace('.jpg', '.png'))
        pm_path = os.path.join(pm_dir, im_name.replace('.jpg', '.png'))

        cm = cv2.imread(cm_path, 0)
        pm = cv2.imread(pm_path, 0)
        print(np.unique(cm))
        if cm is None or pm is None:
            continue

        pm = pm / 255
        bad = (pm < 0.5)

        for i in np.unique(cm):
            mask = (cm == i)
            mask[bad] = 0
            cat = i + 1

            ann = make_ann(mask, cat)
            ann["image_id"] = imgId
            ann["id"] = len(annotations)
            annotations.append(ann)
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--im_dir', type=str, default="../data/ade20k/images/")
    parser.add_argument('-l', '--im_list', type=str, default="../data/ade20k/images/validation.txt")
    parser.add_argument('-o', '--out_dir', type=str, default="../data/ade20k/predictions/pspnet/")
    args = parser.parse_args()
    
    cat_list = get_ade150_dataset()
    im_list = []
    with open(args.im_list,'r') as f:
        im_list = f.read().splitlines()
    print(len(im_list))

    images = make_images(args.im_dir, im_list)
    categories = make_categories(cat_list)
    annotations = make_annotations(args.out_dir, im_list)

    out_file = os.path.join(args.out_dir, "predictions.json")
    save_ann_fn(images, annotations, categories, out_file)





