import os
import cv2
import json
import argparse
import re
import numpy as np
from tqdm import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

def make_images(im_list, im_dir):
    images = []
    for im_name in tqdm(im_list):
        img = {}
        img["file_name"] = im_name
        img["id"] = len(images) + 1

        im_path = os.path.join(im_dir, im_name)
        im = cv2.imread(im_path)
        img["height"] = im.shape[0]
        img["width"] = im.shape[1]
        
        images.append(img)
    return images

def make_categories(cat_list):
    categories = []
    cat_list.remove("__background__")
    for name in cat_list:
        categories.append({"id": len(categories) + 1, "name": name})
    return categories

def make_annotations(ann_list):
    annotations = []
    for a in tqdm(ann_list):
        ann = {}
        ann["id"] = len(annotations) + 1
        ann["category_id"] = a["category_id"]
        ann["image_id"] = a["image_id"]
        ann["iscrowd"] = 0
        if "score" in a:
            ann["score"] = a["score"]

        segm = a["segmentation"]
        ann["segmentation"] = segm
        ann["area"] = int(COCOmask.area(segm))
        ann["bbox"] = list(COCOmask.toBbox(segm))
        annotations.append(ann)
    return annotations

def make_ann(mask, iscrowd=0):
    mask = np.asfortranarray(mask)
    mask = mask.astype(np.uint8)
    segm = COCOmask.encode(mask)
    segm["counts"] = segm["counts"].decode('ascii')

    ann = {}
    ann["segmentation"] = segm
    ann["bbox"] = list(COCOmask.toBbox(segm))
    ann["area"] = int(COCOmask.area(segm))
    ann["iscrowd"] = int(iscrowd)
    return ann

def save_ann_fn(images, annotations, categories, out_fn, indent=2):
    ann_fn = {}
    ann_fn["images"] = images
    ann_fn["annotations"] = annotations
    ann_fn["categories"] = categories

    dirname = os.path.dirname(out_fn)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    
    with open(out_fn, 'w') as f:
        json.dump(ann_fn, f, indent=indent)

def print_ann_fn(ann_fn, show_examples=False):
    if isinstance(ann_fn, str):
        print("File name:", ann_fn)
        coco = COCO(ann_fn)
    elif isinstance(ann_fn, COCO):
        coco = ann_fn

    imgs_with_anns = set([coco.anns[annId]["image_id"] for annId in coco.anns])
    print("Images: {}, Annotations: {}, Categories: {}, Images with anns: {}".format(len(coco.imgs), len(coco.anns), len(coco.cats), len(imgs_with_anns)))

    if "score" in coco.dataset["annotations"][0]:
        thresholds = [0.8, 0.6, 0.4, 0.2, 0]
        counts = []
        for t in thresholds:
            anns = [ann for ann in coco.dataset["annotations"] if ann["score"] > t]
            counts.append(len(anns))
        print("Num of anns with score:", thresholds, counts)

    counts = {}
    for catId in coco.cats:
        catName = coco.cats[catId]["name"]
        annIds = coco.getAnnIds(catIds=[catId])
        counts[catName] = len(annIds)
    print("Counts:", counts)

    if show_examples:
        print("Image examples:")
        for img in coco.dataset["images"][:2]:
            print(img)
        print("Annotation examples:")
        for ann in coco.dataset["annotations"][:2]:
            print(ann)
        print("Category examples:")
        for cat in coco.dataset["categories"][:2]:
            print(cat)

def read_list(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()

def read_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    print_ann_fn(args.ann_fn, show_examples=True)
