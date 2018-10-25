import os
import numpy as np
import cv2

def make_annotations(cm, pm):
    pass

def make_ann_fn(im_list, cat_list, im_dir, output_dir):
    images = []
    annotations = []
    categories = []

    for imgId, im_name in enumerate(im_list):
        im_path = os.path.join(im_dir, im_name)
        cm_path = os.path.join(cm_dir, im_name)
        pm_path = os.path.join(pm_dir, im_name)

        im = cv2.imread(im_path)
        cm = cv2.imread(cm_path)
        pm = cv2.imread(pm_path)

        # Images
        img = {}
        img["file_name"] = im_name
        img["id"] = imgId
        img["height"] = im.shape[0]
        img["width"] = im.shape[1]
        images.append(img)

        # Annotations
        anns = make_annotations(cm, pm)
        for ann in anns:
            ann["image_id"] = imgId
            ann["id"] = len(annotations)
            annotations.append(ann)

    # Categories
    for i, name in enumerate(cat_list):
        categories.append({"id": i, "name": name})

    ann_fn = {}
    ann_fn["images"] = images
    ann_fn["annotations"] = annotations
    ann_fn["categories"] = categories
    return ann_fn


