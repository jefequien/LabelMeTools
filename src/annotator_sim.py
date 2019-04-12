import os
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *

splitB_gt = "/data/vision/torralba/ade20k-places/data/annotation/ade_challenge/ann_files/b_split.json"

class SimulatedAnnotator:

    def __init__(self):
        self.cocoGt = COCO(splitB_gt)
        self.thresholdIOU = 0.8
        self.filenameToImg = {}
        self.nameToCat = {}

        self.setup()

    def setup(self):
        for imgId in coco.imgs:
            img = coco.imgs[imgId]
            self.filenameToImg[img["file_name"]] = img

        for catId in coco.cats:
            cat = coco.cats[catId]
            self.nameToCat[cat["name"]] = cat

    def annotate(self, cocoDt):
        passed_annotations = []

        for dt_imgId in tqdm(cocoDt.imgs):
            dt_img = coco.imgs[dt_imgId]
            dt_annIds = coco.getAnnIds(imgIds=[dt_imgId])
            dt_anns = coco.loadAnns(dt_annIds)
            for dt_ann in dt_anns:
                dt_cat = cocoDt.cats[dt_ann["category_id"]]
                dts = [dt_ann["segmentation"]]

                gt_cat = self.nameToCat[dt_cat["name"]]
                gt_img = self.filenameToImg[dt_img["file_name"]]
                gt_annIds = self.cocoGt.getAnnIds(imgIds=[gt_img["id"]], catIds=[gt_cat["id"]])
                gt_anns = self.cocoGt.loadAnns(gt_annIds)
                gts = [ann["segmentation"] for ann in gt_anns]
                if len(gts) == 0:
                    continue

                iscrowds = [0 for _ in gts]
                ious = COCOmask.iou(dts, gts, iscrowds)

                max_dt_ious = np.max(ious, axis=1)
                max_dt_gtIds = np.argmax(ious, axis=1)
                for dt_ann, max_dt_iou, max_dt_gtId in zip(dt_anns, max_dt_ious, max_dt_gtIds):
                    if max_dt_iou >= self.thresholdIOU:
                        passed_annotations.append(dt_ann)
        return passed_annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, required=True, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_annotated.json")
    print(args)

    coco = COCO(args.ann_fn)
    annotator = SimulatedAnnotator()
    passed = annotator.annotate(coco)

    images = coco.dataset["images"]
    annotations = passed
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(args.out_fn)

