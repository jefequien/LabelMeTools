import argparse
import os
import json
import numpy as np
import cv2
from tqdm import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

def make_images(im_list, im_dir):
    images = []
    for i, im_name in tqdm(enumerate(im_list)):
        img = {}
        img["file_name"] = im_name
        img["id"] = i + 1

        im_path = os.path.join(im_dir, im_name)
        # im = cv2.imread(im_path)
        # img["height"] = im.shape[0]
        # img["width"] = im.shape[1]
        
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

def save_ann_fn(images, annotations, categories, out_fn):
    ann_fn = {}
    ann_fn["images"] = images
    ann_fn["annotations"] = annotations
    ann_fn["categories"] = categories

    dirname = os.path.dirname(out_fn)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    
    with open(out_fn, 'w') as f:
            json.dump(ann_fn, f, indent=2)

def print_ann_fn(ann_fn):
    coco = COCO(ann_fn)
    print("File name:", ann_fn)
    print("Images:", len(coco.imgs))
    print("Annotations:", len(coco.anns))
    print("Categories:", len(coco.cats))

    counts = {}
    for cat in coco.cats:
        catName = coco.cats[cat]["name"]
        annIds = coco.getAnnIds(catIds=[cat])
        counts[catName] = len(annIds)
    print("Counts:", counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str)
    args = parser.parse_args()

    open_coco(args.input_file)
