import json
from pycocotools.coco import COCO

def open_coco(ann_fn):
    coco = COCO(ann_fn)
    for n,id in enumerate(coco.imgs):
        print(coco.imgs[id])
        if n > 10:
            break

    for n,id in enumerate(coco.anns):
        print(coco.anns[id])
        if n > 10:
            break
    for n,id in enumerate(coco.cats):
        print(coco.cats[id])


ade_ann_fn = "../data/ade20k/annotations/instances_ade20k_val.json"
coco_ann_fn = "../data/coco/annotations/instances_train2017.json"
open_coco(ade_ann_fn)
open_coco(coco_ann_fn)

