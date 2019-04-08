import os
import argparse

from coco_utils.coco_format import *
from coco_utils.dummy_datasets import *

def read_list(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
        return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--im_dir', type=str, help='Images directory')
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default="../coco.json", help='Output coco file')

    parser.add_argument('-i', '--im_list', type=str, help='List of images')
    parser.add_argument('-a', '--ann_list', type=str, help='List of annotations. Typically the output of maskrcnn')
    parser.add_argument('-c', '--cat_list', type=str, help='List of categories')
    args = parser.parse_args()
    print(args)

    coco = COCO(args.ann_fn)
    if not args.ann_fn:
        coco.dataset["images"] = []
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = []

    # Make categories
    categories = coco.dataset["categories"]
    if args.cat_list:
        cat_list = []
        if args.cat_list == "coco":
            cat_list = get_coco_dataset()
        elif args.cat_list == "ade100":
            cat_list = get_ade100_dataset()
        elif args.cat_list == "ade150":
            cat_list = get_ade150_dataset()
        else:
            cat_list = read_list(args.cat_list)

        categories = make_categories(cat_list)

    # Make annotations
    annotations = coco.dataset["annotations"]
    if args.ann_list:
        annotations = json.load(args.annotations)

    # Make images
    images = coco.dataset["images"]
    if args.im_list:
        im_list = read_list(args.im_list)
        images = make_images(im_list, args.im_dir)

    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(args.out_fn)
