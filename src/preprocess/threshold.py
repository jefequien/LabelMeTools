import os
import argparse
import json
import numpy as np

from pycocotools.coco import COCO
from coco_format import save_ann_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    args = parser.parse_args()

    coco = COCO(args.input_file)
    thresholded = []
    c = 0
    for annId in coco.anns:
        ann = coco.anns[annId]
        if ann["score"] > args.threshold:
            thresholded.append(ann)
        c += 1
        print("{}/{} {}".format(c, len(coco.anns), len(thresholded)))

    out_fn = os.path.basename(args.input_file).replace('.json', '_{}.json'.format(args.threshold))
    out_file = os.path.join(args.output_dir, out_fn)

    images = list(coco.imgs.values())
    annotations = thresholded
    categories = list(coco.cats.values())
    save_ann_fn(images, annotations, categories, out_file)
