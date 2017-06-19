import argparse
import os
import json
from skimage import io

from maskToPolygons import MaskToPolygons
import utils

categories = utils.get_categories()

def createPolygons(category_mask):
    converter = MaskToPolygons()
    categoryToPolygons, debug = converter.process(category_mask)

    data = {}
    for cat in categoryToPolygons:
        counter = 0
        for polygon in categoryToPolygons[cat]:
            key = "{}#{}".format(categories[cat], counter)
            data[key] = polygon
            counter += 1
    return data

parser = argparse.ArgumentParser()
parser.add_argument("--list", help="Images list")
parser.add_argument("--category_mask", help="Root category mask folder")
parser.add_argument("--polygons", default="./polygons", help="Output polygons folder")
args = parser.parse_args()


# root_category_mask = "/data/vision/torralba/scratch2/hangzhao/movie/pspnet_prediction/category_mask/"
# root_polygons = "/data/vision/oliva/scenedataset/scaleplaces/movie/polygons"
# txt_imlist = "/data/vision/torralba/scratch2/hangzhao/movie/images/images.txt"

root_category_mask = args.category_mask
root_polygons = args.polygons
txt_imlist = args.list

imlist = [line.rstrip() for line in open(txt_imlist, 'r')]

counter = 0
for txt_im in imlist:
    if counter % 1000 == 0:
        print txt_im
    counter += 1

    category_mask_name = txt_im.replace(".jpg", ".png")
    category_mask = io.imread(os.path.join(root_category_mask, category_mask_name), as_grey=True)
    data = createPolygons(category_mask)

    polygon_file = txt_im.replace(".jpg", "-polygons.json")
    polygon_file_path = os.path.join(root_polygons, polygon_file)

    if not os.path.exists(os.path.dirname(polygon_file_path)):
        os.makedirs(os.path.dirname(polygon_file_path))
    with open(polygon_file_path, 'w') as f:
        json.dump(data, f)

