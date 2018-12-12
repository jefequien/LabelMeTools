import os
import json
import numpy as np

from changeRLE import maskToRLE

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

IMAGES_URL = "http://places.csail.mit.edu/scaleplaces/datasets/"

def make_task(imgId, annIds, coco):
    file_name = coco.imgs[imgId]["file_name"]

    annotations = []
    anns = coco.loadAnns(annIds);
    for ann in anns:
        rle = ann['segmentation']
        mask = COCOmask.decode(rle)

        ann = {}
        ann["name"] = coco.cats[ann['category_id']]['name']
        ann["segmentation"] = maskToRLE(mask)
        annotations.append(ann)

    task = {}
    task["image_url"] = img_url
    task["annotations"] = annotations
    return task

def get_image_dir(dataset_name):
    if "ade" in dataset_name:
        return "ade20k/images"
    elif "coco" in dataset_name:
        return "coco/images"
    elif "places" in dataset_name:
        return "places/images"
    else:
        raise Exception("Dataset not implemented")


if __name__ == "__main__":
    im_dir = os.path.join(IMAGES_URL, "ade20k/images")
    ann_fn = "ade20k_train_annotations.json"
    coco = COCO(ann_fn)

    tasks = []
    for i in range(10):
        img = coco.imgs[i]
        annIds = coco.getAnnIds(imgIds=[img['id']])
        anns = coco.loadAnns(ids=annIds)
        img_url = os.path.join(im_dir, img['file_name'])
        
        task = make_task(img_url, anns, coco)
        task['task_id'] = i

        tasks.append(task)
        
    json_repr = json.dumps(tasks, cls=MyEncoder, sort_keys=True, indent=2)
    with open('../tasks/tasks.json', 'w') as outfile:
        outfile.write(json_repr)