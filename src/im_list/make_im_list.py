import os
import argparse
from tqdm import tqdm

def make_im_list(im_dir, prefix):
    im_list = []
    for root, dirs, files in tqdm(os.walk(im_dir)):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == ".jpg" or ext == ".png":
                file_name = os.path.join(root, file)
                im_name = os.path.relpath(file_name, prefix)
                im_list.append(im_name)
    return im_list

def write_im_list(im_list, out_fn):
    out_str = "\n".join(im_list) + "\n"
    with open(out_fn, 'w') as f:
        f.write(out_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--im_dir', type=str, help='Images directory')
    parser.add_argument('-p', '--prefix', type=str, help='Prefix to remove. Default is equal to im_dir')
    args = parser.parse_args()
    if args.prefix == None:
        args.prefix = args.im_dir
    print(args)

    im_list = make_im_list(args.im_dir, args.prefix)
    out_fn = "im_list.txt"
    write_im_list(im_list, out_fn)

