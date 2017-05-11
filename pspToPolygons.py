import argparse
import os
import json
from skimage import io

from maskToPolygons import MaskToPolygons

psp_path = "/data/vision/torralba/scratch2/hangzhao/scale_places/pspnet_prediction"
all_polygons_path = "/data/vision/oliva/scenedataset/scaleplaces/LabelMe_data/polygons"

psp_path_sample = "./pspnet_sample"
polygons_sample = "./polygons"

# Comment out before running
psp_path = psp_path_sample
all_polygons_path = polygons_sample

converter = MaskToPolygons()

def listOptions(c):
    path = "{}/category_mask/{}".format(psp_path,c)
    print [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def process(category_mask, prob_mask, max_prob):
    categoryToPolygons, debug = converter.processMask(category_mask)

    data = {}
    for category in categoryToPolygons:
        counter = 0
        for polygon in categoryToPolygons[category]:
            key = "{}#{}".format(category, counter)
            # Flip (h,w) to (x,y)
            flipped = [[point[1], point[0]] for point in polygon]
            data[key] = polygon
    return data



parser = argparse.ArgumentParser()
parser.add_argument("--list", help="List categories for letter")
parser.add_argument("--process", help="Category folder you want to preprocess")
args = parser.parse_args()

if args.list:
    listOptions(args.list)
else:
    category = args.process

    # Create folder
    polygons_path = "{}/{}/{}".format(all_polygons_path, category[0], category)
    if not os.path.exists(polygons_path):
        os.makedirs(polygons_path)

    category_mask_path = "{}/category_mask/{}/{}".format(psp_path, category[0], category)
    max_prob_path = "{}/max_prob/{}/{}".format(psp_path, category[0], category)
    prob_mask_path = "{}/prob_mask/{}/{}".format(psp_path, category[0], category)
    
    files = sorted(os.listdir(category_mask_path))
    print "Processing {} images".format(len(files))
    for f in files:
        fname = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1]
        if ext == ".png":
            print fname

            category_mask = io.imread("{}/{}.png".format(category_mask_path, fname), as_grey=True)
            prob_mask = io.imread("{}/{}.jpg".format(prob_mask_path, fname), as_grey=True)
            max_prob = "{}/{}.h5".format(max_prob_path, fname)

            data = process(category_mask, prob_mask, max_prob)
            polygon_file = "{}/{}-polygons.json".format(polygons_path, fname)
            with open(polygon_file, 'w') as outfile:
                json.dump(data, outfile)
