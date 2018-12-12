import argparse
import os
import cv2
import numpy as np

def get_im_list(im_dir):
    im_list = []
    for root, dirs, files in os.walk(im_dir):
        for name in files:
            name = os.path.join(root, name)
            im_list.append(name)
    print(len(im_list))
    im_list.sort()
    im_list = im_list[:100]
    return im_list

def write_video(im_list, outdir):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(outdir, 'output.mp4'),fourcc, 3.0, (2048, 1024))

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
    write_video(im_list, args.outdir)



