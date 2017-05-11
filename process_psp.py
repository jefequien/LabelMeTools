import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--category", help="Category folder you want to preprocess")
args = parser.parse_args()

category = args.category

psp_path = "/data/vision/torralba/scratch2/hangzhao/scale_places/pspnet_prediction"
category_mask_path = "{}/category_mask/{}/{}".format(psp_path, category[0], category)
max_prob_path = "{}/max_prob/{}/{}".format(psp_path, category[0], category)
prob_mask_path = "{}/prob_mask/{}/{}".format(psp_path, category[0], category)

for f in os.listdir(category_mask_path):
    print f


print category_mask_path
print max_prob_path
print prob_mask_path
