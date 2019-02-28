import argparse
import os
import json
import numpy as np
import cv2

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

def make_images(im_list, im_dir):
    print("Opening images...")
    images = []
    for i, im_name in enumerate(im_list):
        img = {}
        img["file_name"] = im_name
        img["id"] = i + 1

        im_path = os.path.join(im_dir, im_name)
        im = cv2.imread(im_path)
        img["height"] = im.shape[0]
        img["width"] = im.shape[1]
        
        images.append(img)
    return images

def make_categories(cat_list):
    categories = []
    cat_list.remove("__background__")
    for i, name in enumerate(cat_list):
        categories.append({"id": i + 1, "name": name})
    return categories

def make_ann(mask, iscrowd=0):
    mask = np.asfortranarray(mask)
    mask = mask.astype(np.uint8)
    segm = COCOmask.encode(mask)
    segm["counts"] = segm["counts"].decode('ascii')

    ann = {}
    ann["segmentation"] = segm
    ann["area"] = int(COCOmask.area(segm))
    ann["bbox"] = list(COCOmask.toBbox(segm))
    ann["iscrowd"] = int(iscrowd)
    return ann

def save_ann_fn(images, annotations, categories, out_file):
    ann_fn = {}
    ann_fn["images"] = images
    ann_fn["annotations"] = annotations
    ann_fn["categories"] = categories

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
        
    with open(out_file, 'w') as f:
            json.dump(ann_fn, f, indent=2)

def open_coco(ann_fn):
    coco = COCO(ann_fn)
    print(len(coco.imgs), "images")
    print(len(coco.anns), "annotations")
    print(len(coco.cats), "categories")
    for n, id in enumerate(coco.imgs):
        print(coco.imgs[id])
        if n > 10:
            break

    for n, id in enumerate(coco.anns):
        print(coco.anns[id])
        if n > 10:
            break
    for n, id in enumerate(coco.cats):
        print(coco.cats[id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str)
    args = parser.parse_args()

    open_coco(args.input_file)