import os
import sys
sys.path.append("../coco_utils")
import random
import argparse
import requests
import json
from tqdm import tqdm

from pycocotools.coco import COCO
from coco_format import *

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
    print("Splitting into bundles...")
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

def post_bundles(job_id, bundles, base_url):
    print("Posting bundles to {}".format(base_url))
    status_codes = {}
    bundle_ids = []
    endpoint = base_url + "/api/bundles?job_id=" + job_id;
    for bundle in tqdm(bundles):
        r = requests.post(url=endpoint, json=bundle)
        if r.status_code == 200:
            res = json.loads(r.text)
            bundle_ids.append(res["bundle_id"])

        if r.status_code not in status_codes:
            status_codes[r.status_code] = 0
        status_codes[r.status_code] += 1

    print("Status Codes:", status_codes)
    return bundle_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', type=str, required=True)
    parser.add_argument('-f', '--ann_fn', type=str, required=True)
    parser.add_argument('-p', '--prod', action='store_true')
    args = parser.parse_args()
    print(args)

    coco = COCO(args.ann_fn)

    base_url = "http://localhost:3000"
    if args.prod:
        base_url = "https://labelmelite.csail.mit.edu"

    bundles = split_into_bundles(coco)
    bundle_ids = post_bundles(args.job_id, bundles, base_url)

    write_list("{}.txt".format(args.job_id), bundle_ids)

