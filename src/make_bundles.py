import os
import argparse
import uuid
import random
import numpy as np
from tqdm import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from coco_utils.coco_format import save_ann_fn

# Bundles are Coco annotation files

def threshold(coco, t):
    annotations = []
    for ann in coco.dataset["annotations"]:
        if ann["score"] >= t:
            annotations.append(ann)

    coco.dataset["annotations"] = annotations
    coco.createIndex()

def make_bundle(coco, annotations, bundle_dir):
    imgIds = set()
    catIds = set()
    for ann in annotations:
        imgIds.add(ann["image_id"])
        catIds.add(ann["category_id"])

    images = [coco.imgs[id] for id in imgIds]
    categories = [coco.cats[id] for id in catIds]

    bundle_id = uuid.uuid4().hex
    out_fn = os.path.join(bundle_dir, "{}.json".format(bundle_id))
    save_ann_fn(images, annotations, categories, out_fn)

def split_into_bundles(coco, bundle_dir, NUM=30):
    # Sort annotations by category
    anns_sorted = []
    for catId in coco.cats:
        annIds_cat = coco.getAnnIds(catIds=[catId])
        anns_cat = coco.loadAnns(annIds_cat)
        random.shuffle(anns_cat)
        anns_sorted.extend(anns_cat)

    anns_split = [anns_sorted[i:i + NUM] for i in range(0, len(anns_sorted), NUM)]
    for anns in tqdm(anns_split):
        make_bundle(coco, anns, bundle_dir)

def make_bundle_for_category(coco, catId, out_file):
    images = []
    annotations = []
    categories = [coco.cats[catId]]
    imgIds = set()

    for annId in coco.anns:
        ann = coco.anns[annId]
        if ann["category_id"] == catId:
            annotations.append(ann)
            imgIds.add(ann["image_id"])

    for imgId in imgIds:
        images.append(coco.imgs[imgId])

    print("{}: {} annotations, {} images".format(coco.cats[catId]["name"], len(annotations), len(images)))
    save_ann_fn(images, annotations, categories, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ann_fn', type=str, default="../../LabelMe-Lite/data/places/maskrcnna_000.json")
    parser.add_argument('-o', '--out_dir', type=str, default="./bundles")
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    threshold(coco, 0.5)
    split_into_bundles(coco, args.out_dir)



