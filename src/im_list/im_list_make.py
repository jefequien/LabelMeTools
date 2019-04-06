import os
import random
import argparse
from tqdm import tqdm
from glob import glob

def make_im_list(glob_path, prefix, toDelete=None):
    paths = glob(args.glob_path)
    if toDelete == None:
        toDelete = prefix

    im_list = []
    for path in tqdm(paths):
        if prefix in path:
            im_name = path.replace(toDelete, "")

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
    parser.add_argument('-g', '--glob_path', type=str, default="../data/ade20k/images/*/*.jpg", help='Glob path')
    parser.add_argument('-p', '--prefix', type=str, help='Required prefix')
    parser.add_argument('-d', '--delete', type=str, help='To delete if different from prefix')
    parser.add_argument('-m', '--max_num', type=int, default=None, help='Max number of images per im_list')
    parser.add_argument('-r', '--randomize')
    args = parser.parse_args()
    print(args)

    im_list = make_im_list(args.glob_path, args.prefix, args.delete)
    if args.randomize:
        random.seed(1)
        random.shuffle(im_list)

    im_lists = [im_list]
    if args.max_num != None:
        im_lists = split_im_list(im_list, args.max_num)

    for i, im_list in enumerate(im_lists):
        out_fn = "images{}.txt".format(i)
        print("{} / {}".format(i+1, len(im_lists)), out_fn)
        write_im_list(im_list, out_fn)

