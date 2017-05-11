import argparse
import os
from skimage import io

psp_path = "/data/vision/torralba/scratch2/hangzhao/scale_places/pspnet_prediction"

def listDir(c):
	path = "{}/category_mask/{}".format(psp_path,c)
	print os.listdir(path)

parser = argparse.ArgumentParser()
parser.add_argument("--list", help="List categories for letter")
parser.add_argument("--category", help="Category folder you want to preprocess")
args = parser.parse_args()

c = args.list
category = args.category

if c:
	listDir(c)
if category:
	category_mask_path = "{}/category_mask/{}/{}".format(psp_path, category[0], category)
	max_prob_path = "{}/max_prob/{}/{}".format(psp_path, category[0], category)
	prob_mask_path = "{}/prob_mask/{}/{}".format(psp_path, category[0], category)
	
	files = sorted(os.listdir(category_mask_path))
	print "Processing {} images".format(len(files))
	for f in files:
		fname = os.path.basename(f)
		category_mask = io.imread("{}/{}.png".format(category_mask_path, fname), as_grey=True)
                prob_mask = io.imread("{}/{}.jpg".format(prob_mask_path, fname), as_grey=True)
                max_prob = "{}/{}.h5".format(max_prob_path, fname)
