import os
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *
from coco_utils.dummy_datasets import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default="../coco.json", help='Output coco file')

    parser.add_argument('-i', '--im_list', type=str, help='List of images')
    parser.add_argument('-a', '--ann_list', type=str, help='List of annotations. Typically the output of maskrcnn')
    parser.add_argument('-c', '--cat_list', type=str, help='List of categories')
    parser.add_argument('-d', '--im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    args = parser.parse_args()
    print(args)