import argparse
import os
import shutil

places2_path = "/data/vision/oliva/scenedataset/places2new/challenge2016/data_large"
all_polygons_path = "/data/vision/oliva/scenedataset/scaleplaces/LabelMe_data/polygons"
all_trees_path = "/data/vision/oliva/scenedataset/scaleplaces/LabelMe_data/trees"
labelme_dataset_path = "../LabelMe/datasets"

places2_sample = os.path.abspath("./places2_sample")
polygons_sample = os.path.abspath("./polygons")
trees_sample = os.path.abspath("./trees")

# Comment out before running for real
# places2_path = places2_sample
# all_polygons_path = polygons_sample
# all_trees_path = trees_sample

def listOptions(c):
    if len(c) == 1:
        path = "{}/{}".format(places2_path,c)
    elif c == "all":
        path = places2_path

    if os.path.isdir(path):
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    else:
        return []

def link(category):
    unlink(category)
    image_src = "{}/{}/{}".format(places2_path, category[0], category)
    if not os.path.exists(image_src):
        return False

    dataset_path = "{}/{}".format(labelme_dataset_path, category)
    annotations_path = "{}/annotations".format(dataset_path)
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    # Symlink images
    image_dst = "{}/images".format(dataset_path)
    os.symlink(image_src, image_dst)

    # Symlink polygons
    polygons_src = "{}/{}/{}".format(all_polygons_path, category[0], category)
    if not os.path.exists(polygons_src):
        os.makedirs(polygons_src)
    polygons_dst = "{}/polygons".format(annotations_path)
    os.symlink(polygons_src, polygons_dst)

    # Symlink trees
    trees_src = "{}/{}/{}".format(all_trees_path, category[0], category)
    if not os.path.exists(trees_src):
        os.makedirs(trees_src)
    trees_dst = "{}/trees".format(annotations_path)
    os.symlink(trees_src, trees_dst)
    return True


def unlink(category):
    path = "{}/{}".format(labelme_dataset_path, category)
    if category == "examples" or not os.path.exists(path):
        return False
    else:
        shutil.rmtree(path)
        return True


parser = argparse.ArgumentParser()
parser.add_argument("--list", help="List categories for letter. Type all to see all options")
parser.add_argument("--link", help="Category you want to link to LabelMe")
parser.add_argument("--unlink", help="Category you want to unlink to LabelMe")
args = parser.parse_args()

if args.list:
    print listOptions(args.list)
elif args.link:
    if len(args.link) == 1:
        categories = listOptions(args.link)
    else:
        categories = [args.link]
    
    for category in categories:
        success = link(category)
        if success:
            print "Linked {}".format(category)
        else:
            print "Could not find {} image source".format(category)

elif args.unlink:
    category = args.unlink
    success = unlink(category)
    if success:
        print "Unlinked {}".format(category)
    else:
        print "Could not find {} in LabelMe".format(category)