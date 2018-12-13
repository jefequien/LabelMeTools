import argparse
import os
import cv2
import numpy as np

def get_im_list(im_dir):
    im_list = []
    for root, dirs, files in os.walk(im_dir):
        for name in files:
            if '.jpg' in name or '.png' in name:
                name = os.path.join(root, name)
                im_list.append(name)
    im_list.sort()
    print(len(im_list), "images")
    return im_list

def write_video(im_list, out_fn):
    im = cv2.imread(im_list[0])
    h = im.shape[0]
    w = im.shape[1]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, 3.0, (w, h))

    for i,im_name in enumerate(im_list):
        im = cv2.imread(im_name)
        print(i, im_name, im.shape)
        out.write(im)

        # cv2.imshow('frame',im)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str)
    parser.add_argument('-o', '--outdir', type=str)
    args = parser.parse_args()

    im_list = get_im_list(args.indir)
    out_fn = os.path.basename(os.path.normpath(args.indir))
    out_fn = os.path.join(args.outdir, '{}.mp4'.format(out_fn))
    print(out_fn)

    write_video(im_list, out_fn)
    write_video(im_list[:100], out_fn.replace('.mp4', '_short.mp4'))

