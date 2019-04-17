import os
import sys
sys.path.append('../coco_utils')
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

from coco_format import *

def assocatiate_keypoints(anns):
    kp_only_anns = []
    kp_missing_anns = []
    other_anns = []
    for ann in anns:
        if "keypoints" in ann and "segmentation" not in ann:
            kp_only_anns.append(ann)
        elif "keypoints" not in ann:
            kp_missing_anns.append(ann)
        else:
            other_anns.append(ann)

    for ann in kp_missing_anns:
        mask = COCOmask.decode(ann["segmentation"])
        max_good_kps = 0
        for keyp_ann in kp_only_anns:
            if ann["category_id"] != keyp_ann["category_id"]:
                continue

            # Count number of keypoints to lie in the mask
            num_good_kps = 0
            num_kps = 0
            kps = np.array(keyp_ann["keypoints"]).reshape(-1, 3)
            for x,y,v in kps:
                if v != 0:
                    num_kps += 1
                    if mask[int(y), int(x)] != 0:
                        num_good_kps += 1
            if num_good_kps > max_good_kps:
                ann["keypoints"] = keyp_ann["keypoints"]
                ann["num_keypoints"] = num_kps
                max_good_kps = num_good_kps
        
        if max_good_kps > 0:
            other_anns.append(ann)
    return other_anns

def print_annotation_types(anns):
    skp_anns = [ann for ann in anns if "keypoints" in ann and "segmentation" in ann]
    s_anns = [ann for ann in anns if "keypoints" not in ann and "segmentation" in ann]
    kp_anns = [ann for ann in anns if "keypoints" in ann and "segmentation" not in ann]
    print("{} S+KP, {} S, {} KP annotations".format(len(skp_anns), len(s_anns), len(kp_anns)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_associated.json")
    print(args)

    coco = COCO(args.ann_fn)
    print_annotation_types(coco.dataset["annotations"])

    # Associate keypoints
    annotations = []
    for imgId in tqdm(coco.imgs):
        annIds = coco.getAnnIds(imgIds=[imgId])
        anns = coco.loadAnns(annIds)
        new_anns = assocatiate_keypoints(anns)
        annotations.extend(new_anns)

    print_annotation_types(annotations)

    images = coco.dataset["images"]
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(args.out_fn)
