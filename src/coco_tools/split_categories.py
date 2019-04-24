import os
import sys
sys.path.append("../coco_utils")
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from coco_format import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = args.ann_fn.replace(".json", "_categories")
    print(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    coco = COCO(args.ann_fn)
    for catId in tqdm(coco.cats):
        cat = coco.cats[catId]
        categories = [cat]
        annotations = [ann for ann in coco.dataset["annotations"] if ann["category_id"] == catId]
        imgIds = set([ann["image_id"] for ann in annotations])
        images = [coco.imgs[imgId] for imgId in imgIds]

        out_fn = os.path.join(args.out_dir, "{}_{}.json".format(catId, cat["name"]))
        save_ann_fn(images, annotations, categories, out_fn)
        print_ann_fn(out_fn)
