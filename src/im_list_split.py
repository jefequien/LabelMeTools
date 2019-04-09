import os
import argparse
import random
from tqdm import tqdm

def read_list(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()

def write_list(im_list, out_fn):
    out_str = "\n".join(im_list) + "\n"
    with open(out_fn, 'w') as f:
        f.write(out_str)

def split_im_list(im_list, n):
    c = 0
    im_lists = []
    while c < len(im_list):
        im_lists.append(im_list[c:c+n])
        c += n
    return im_lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--im_list', type=str, help='List of images')
    parser.add_argument('-n', '--num', type=int, help='Number of images to split per image list')
    args = parser.parse_args()
    print(args)
    
    im_list = read_list(args.im_list)
    im_lists = split_im_list(im_list, args.num)

    for i, im_list in tqdm(enumerate(im_lists)):
        out_fn = args.im_list.replace(".txt", "{}.txt".format(i.zfill(3)))
        write_im_list(im_list, out_fn)
        