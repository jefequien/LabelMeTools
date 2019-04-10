import os
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *

def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-a', '--ann_list', type=str, help='List of annotations. The output of maskrcnn')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_pred.json")
    print(args)

    coco = COCO(args.ann_fn)
    ann_list = load_json(args.ann_list)

    images = coco.dataset["images"]
    annotations = make_annotations(ann_list)
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(out_fn)
