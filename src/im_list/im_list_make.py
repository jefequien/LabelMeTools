import os
import random
import argparse
from glob import glob

def make_im_list(im_dir, glob_path):
    paths = glob(args.glob_path)
    im_list = []
    for path in paths:
        im_name = path.replace(im_dir, "")
        if (im_name[0] == '/'):
            im_name = im_name[1:]
        im_list.append(im_name)
    return im_list

def split_im_list(im_list, n):
    c = 0
    im_lists = []
    while c < len(im_list):
        im_lists.append(im_list[c:c+n])
        c += n
    return im_lists

def write_im_list(im_list, out_fn):
    out_str = "\n".join(im_list)
    with open(out_fn,'w') as f:
        f.write(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--im_dir', type=str, default="../data/ade20k/images", help='Root image directory')
    parser.add_argument('-g', '--glob_path', type=str, default="../data/ade20k/images/*/*.jpg", help='Glob path')
    parser.add_argument('-m', '--max_num', type=int, default=None, help='Max number of images per im_list')
    args = parser.parse_args()

    im_list = make_im_list(args.im_dir, args.glob_path)
    random.seed(1)
    random.shuffle(im_list)
    print(len(im_list), "images")
    im_lists = [im_list]
    if args.max_num != None:
        im_lists = split_im_list(im_list, args.max_num)

    for i, im_list in enumerate(im_lists):
        out_fn = "images{}.txt".format(i)
        print("{} / {}".format(i+1, len(im_lists)), out_fn)
        write_im_list(im_list, out_fn)

