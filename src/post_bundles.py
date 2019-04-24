import os
import random
import argparse
import requests
import json
from tqdm import tqdm

from pycocotools.coco import COCO

# Bundles are Coco annotation files
# API_ENDPOINT = "http://localhost:3000/api/bundles"
API_ENDPOINT = "https://labelmelite.csail.mit.edu/api/bundles"

def threshold(coco, t):
    annotations = []
    for ann in coco.dataset["annotations"]:
        if "score" in ann and ann["score"] < t:
            continue
        annotations.append(ann)

    coco.dataset["annotations"] = annotations
    coco.createIndex()

def make_bundle(coco, annotations):
    imgIds = set()
    catIds = set()
    for ann in annotations:
        imgIds.add(ann["image_id"])
        catIds.add(ann["category_id"])

    images = [coco.imgs[id] for id in imgIds]
    categories = [coco.cats[id] for id in catIds]

    bundle = {}
    bundle["images"] = images
    bundle["annotations"] = annotations
    bundle["categories"] = categories
    return bundle

def split_into_bundles(coco, NUM=30):
    # Sort annotations by category
    anns_sorted = []
    for catId in coco.cats:
        annIds_cat = coco.getAnnIds(catIds=[catId])
        anns_cat = coco.loadAnns(annIds_cat)
        random.shuffle(anns_cat)
        anns_sorted.extend(anns_cat)

    bundles = []
    anns_split = [anns_sorted[i:i + NUM] for i in range(0, len(anns_sorted), NUM)]
    for anns in tqdm(anns_split):
        bundle = make_bundle(coco, anns)
        bundles.append(bundle)
    return bundles

def post_bundles(bundles):
    bundle_ids = []
    status_codes = {}
    for bundle in bundles:
        r = requests.post(url=API_ENDPOINT, json=bundle)
        if r.status_code == 200:
            response = json.loads(r.text)
            bundle_info = response["bundle_info"]
            bundle_ids.append(bundle_info["bundle_id"])
            print(bundle_info)

        if r.status_code not in status_codes:
            status_codes[r.status_code] = 0
        status_codes[r.status_code] += 1

    print("Status Codes:", status_codes)
    return bundle_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    threshold(coco, 0.2)

    bundles = split_into_bundles(coco)
    bundle_ids = post_bundles(bundles)



