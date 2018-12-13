import argparse
import os
import cv2

from video_annotation import VideoAnnotation, read_file

def process_video(vid_fn, img_dir):
    cap = cv2.VideoCapture(vid_fn)
    vid_name = os.path.splitext(os.path.basename(vid_fn))[0]
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_num % 10 != 0):
            frame_num += 1
            continue
        print("{}/{}".format(frame_num, length))

        frame_name = "{}_{}.jpg".format(vid_name, frame_num)
        img_path = os.path.join(img_dir, frame_name)
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        cv2.imwrite(img_path, frame)
        frame_num += 1
    cap.release()

def process_annotations(vid_fn, ann_fn, ann_dir):
    cap = cv2.VideoCapture(vid_fn)
    vid_name = os.path.splitext(os.path.basename(vid_fn))[0]
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    vid_ann = VideoAnnotation(ann_fn)
    images = []
    annotations = []
    categories = []

    img_id = 0
    for frame_num in range(0, length, 10):
        objs = vid_ann.get_objects_at_frame(frame_num)
        evts = vid_ann.get_events_at_frame(frame_num)
        for obj in objs:
            ann = {}
            ann["image_id"] = img_id
            ann["id"] = len(annotations)
            annotations.append(ann)
        for evt in evts:
            ann = {}
            ann["image_id"] = img_id
            ann["id"] = len(annotations)
            annotations.append(ann)

        frame_name = "{}_{}.jpg".format(vid_name, frame_num)
        img_id += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="train")
    parser.add_argument('-o', '--outdir', type=str, default="../data/virat/")
    args = parser.parse_args()

    VID_DIR = "../data/virat/raw_data/VIRAT/videos_original"
    if args.split == "train":
        indir = "../data/virat/raw_data/VIRAT-V1_JSON_train-leaderboard_drop4_20180614"
    elif args.split == "val":
        indir = "../data/virat/raw_data/VIRAT-V1_JSON_validate-leaderboard_drop4_20180614"

    file_index = read_file(os.path.join(indir, "file-index.json"))
    vid_list = [os.path.splitext(k)[0] for k in file_index]
    vid_list.sort()

    img_dir = os.path.join(args.outdir, "images", args.split)
    ann_dir = os.path.join(args.outdir, "annotations", args.split)
    for vid_name in vid_list:
        print(vid_name)
        vid_fn = os.path.join(VID_DIR, vid_name + ".mp4")
        ann_fn = os.path.join(indir, vid_name + ".json")

        if not os.path.exists(vid_fn):
            print "Could not load video", vid_fn
            continue

        if not os.path.exists(ann_fn):
            print "Could not load annotations", ann_fn
            continue

        process_video(vid_fn, img_dir)
        # process_annotations(vid_fn, ann_fn, ann_dir)