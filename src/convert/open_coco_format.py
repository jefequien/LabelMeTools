import argparse
import json
from pycocotools.coco import COCO

def open_coco(ann_fn):
    coco = COCO(ann_fn)
    for n, id in enumerate(coco.imgs):
        print(coco.imgs[id])
        if n > 10:
            break

    for n, id in enumerate(coco.anns):
        print(coco.anns[id])
        if n > 10:
            break
    for n, id in enumerate(coco.cats):
        print(coco.cats[id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str)
    args = parser.parse_args()

    open_coco(args.input_file)

