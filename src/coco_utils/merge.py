import os
import argparse
import copy
from tqdm import tqdm

from pycocotools.coco import COCO
from coco_format import *

def merge_cocos(cocos):
    images = []
    annotations = []
    categories = []

    filename_to_id = {}
    catname_to_id = {}

    for coco in tqdm(cocos):
        for ann in coco.dataset["annotations"]:
            img = coco.imgs[ann["image_id"]]
            cat = coco.cats[ann["category_id"]]

            img = copy.deepcopy(img)
            ann = copy.deepcopy(ann)
            cat = copy.deepcopy(cat)
            ann_id = len(annotations) + 1
            img_id = len(images) + 1
            cat_id = len(categories) + 1
            if img["file_name"] in filename_to_id:
                img_id = filename_to_id[img["file_name"]]
            else:
                filename_to_id[img["file_name"]] = img_id
                images.append(img)
            if cat["name"] in catname_to_id:
                cat_id = catname_to_id[cat["name"]]
            else:
                catname_to_id[cat["name"]] = cat_id
                categories.append(cat)

            ann["id"] = ann_id
            img["id"] = img_id
            cat["id"] = cat_id
            ann["image_id"] = img["id"]
            ann["category_id"] = cat["id"]
            annotations.append(ann)

    coco = COCO()
    coco.dataset["images"] = images
    coco.dataset["annotations"] = annotations
    coco.dataset["categories"] = categories
    coco.createIndex()
    return coco

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, required=True)
    parser.add_argument('-o', '--out_fn', type=str)
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = os.path.normpath(args.in_dir) + ".json"
    print(args)

    cocos = []
    for filename in sorted(os.listdir(args.in_dir)):
        if ".json" == os.path.splitext(filename)[1]:
            ann_fn = os.path.join(args.in_dir, filename)
            coco = COCO(ann_fn)
<<<<<<< HEAD
            print_coco(coco)
=======
>>>>>>> 718b0e58a20653deb9ec22514b9a52ad0945f67c
            cocos.append(coco)

    coco = merge_cocos(cocos)
    save_coco(coco, args.out_fn)

    # Verify out_fn
    coco = COCO(args.out_fn)
    print_coco(coco)

