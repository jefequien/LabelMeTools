import argparse
import os
import cv2
from pycocotools.coco import COCO

from visualize import *

COLORS = {}

def visualize_image(img, anns, coco):
    for ann in anns:
        bbox = ann["bbox"]
        name = coco.cats[ann["category_id"]]["name"]
        if name not in COLORS:
            COLORS[name] = random_color()
        color = COLORS[name]

        img = vis_bbox(img, bbox, color=color)
        img = vis_class(img, (bbox[0], bbox[1] - 2), name, color=color)
    return img

def visualize_dataset(im_dir, ann_fn, out_dir):
    coco = COCO(ann_fn)
    for img_id in range(len(coco.imgs)):
        im = coco.imgs[img_id]
        im_name = im["file_name"]
        print(img_id, im_name)

        img = cv2.imread(os.path.join(im_dir, im_name))
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = [coco.anns[ann_id] for ann_id in ann_ids]

        out_fn = os.path.join(out_dir, im_name)
        if not os.path.exists(os.path.dirname(out_fn)):
            os.makedirs(os.path.dirname(out_fn))

        img = visualize_image(img, anns, coco)
        cv2.imwrite(out_fn, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--im_dir', type=str, default="../data/virat/images/")
    parser.add_argument('-a', '--ann_fn', type=str, default="../data/virat/annotations/train/VIRAT_S_000002.json")
    parser.add_argument('-o', '--outdir', type=str, default="../data/virat/vis")
    args = parser.parse_args()

    visualize_dataset(args.im_dir, args.ann_fn, args.outdir)

