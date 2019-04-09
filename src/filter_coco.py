import os
import argparse

from coco_utils.coco_format import *
from coco_utils.dummy_datasets import *

def read_list(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default="../coco.json", help='Output coco file')

    parser.add_argument('-i', '--im_list', type=str, help='List of images')
    args = parser.parse_args()
    print(args)

    images = []
    annotations = []
    categories = []

    coco = COCO(args.ann_fn)
    if args.ann_fn:
        images = coco.dataset["images"]
        annotations = coco.dataset["annotations"]
        categories = coco.dataset["categories"]

    # Filter by image list
    if args.im_list:
        filtered_images = []
        filtered_annotations = []
        im_list = set(read_list(args.im_list))
        for img in images:
            if img["file_name"] in im_list:
                annIds = coco.getAnnIds(imgIds=[img["id"]])
                anns = coco.loadAnns(annIds)
                filtered_images.append(img)
                filtered_annotations.extend(anns)
        images = filtered_images
        annotations = filtered_annotations

    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(args.out_fn)
