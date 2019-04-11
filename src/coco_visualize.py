import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_BLUE = (127, 18, 15)
_RED = (18, 15, 127)
COLOR_MAP = {}

def get_color(name):
    if name not in COLOR_MAP:
        r =  random.randint(0, 255)
        g =  random.randint(0, 255)
        b =  random.randint(0, 255)
        COLOR_MAP[name] = (b,g,r)
    return COLOR_MAP[name]

def vis_mask(img, mask, alpha=0.4, show_border=True, border_thick=1, color=_GREEN):
    """Visualizes a single binary mask."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * np.array(color)

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)

def vis_class(img, pos, class_str, font_scale=0.35, color=_GREEN):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img

def vis_bbox(img, bbox, thick=1, color=_GREEN):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def vis_keypoints(img, keypoints, skeleton, radius=3, thick=2, color=_WHITE):
    """Visualizes keypoints."""
    img = img.astype(np.uint8)

    kps = {}
    # Draw keypoints
    num_keypoints = int(len(keypoints) / 3)
    for i in range(num_keypoints):
        x,y,v = keypoints[3*i], keypoints[3*i+1], keypoints[3*i+2]
        if v != 0:
            cv2.circle(img, (x,y), radius, color, thickness=-1)
            kps[i+1] = (x,y)
    # Draw skeleton
    for l in skeleton:
        if l[0] in kps and l[1] in kps:
            cv2.line(img, kps[l[0]], kps[l[1]], color, thickness=thick)
    return img

def vis_image(coco, img, anns):
    for ann in anns:
        cat = coco.cats[ann["category_id"]]
        name = cat["name"]
        color = get_color(name)

        # Visualize mask
        if "segmentation" in ann:
            mask = coco.annToMask(ann)
            img = vis_mask(img, mask, color=color)

        # Visualize bbox
        if "bbox" in ann:
            bbox = ann["bbox"]
            img = vis_bbox(img, bbox, color=color)

        # Visualize class name
        if "score" in ann:
            name += " %.2f" % ann["score"]
        img = vis_class(img, (bbox[0], bbox[1] - 2), name, color=color)

        # Visualize keypoints
        if "keypoints" in ann:
            keypoints = ann["keypoints"]
            skeleton = cat["skeleton"]
            img = vis_keypoints(img, keypoints, skeleton)
    return img

def vis_coco(coco, im_dir, out_dir):
    html_fn = os.path.join(out_dir, "all_images.html")
    with open(html_fn, "w") as f:

        # Visualize images
        for imgId in tqdm(coco.imgs):
            im = coco.imgs[imgId]
            im_name = im["file_name"]
            img_fn = os.path.join(im_dir, im_name)
            out_fn = os.path.join(out_dir, im_name)
            if not os.path.exists(os.path.dirname(out_fn)):
                os.makedirs(os.path.dirname(out_fn))

            annIds = coco.getAnnIds(imgIds=[imgId])
            anns = coco.loadAnns(annIds)

            # Visualize annotations
            img = cv2.imread(img_fn)
            if img == None:
                print("Warning: Could not find ", img_fn)
                img = np.zeros((im["height"], im["width"], 3))
            img = vis_image(coco, img, anns)
            cv2.imwrite(out_fn, img)

            # Add to html
            tag = "<img src=\"" + im_name +"\" height=\"300\">"
            f.write(tag + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_dir', type=str, default=None, help='Output visualization directory')
    parser.add_argument('-d', '--im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    args = parser.parse_args()
    if args.out_dir == None:
        args.out_dir = args.ann_fn.replace(".json", "")
    print(args)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    coco = COCO(args.ann_fn)
    vis_coco(coco, args.im_dir, args.out_dir)

