"""
@author: Viet Nguyen (viet.nguyen@id-vision.io)
"""
import os
import glob
import cv2
import json
import shutil
import argparse


def get_args():
    parser = argparse.ArgumentParser("""Convert annotation data to the format that yolo needs""")
    parser.add_argument("--path_input", type=str, default="../../Components/DataHandling/idvision")
    parser.add_argument("--path_output", type=str, default="sequences")
    parser.add_argument("--mode", type=str, default="ball", choices=["all", "player", "ball"])
    parser.add_argument("--ratio", type=float, default=0.9, help="part used for training set")
    parser.add_argument("--min_area", type=int, default=0, help="the object's min area threshold for being kept")
    args = parser.parse_args()
    return args


def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


if __name__ == '__main__':
    args = get_args()
    root = args.path_input
    output_path = "datasets/{}_{}".format(args.path_output, args.mode)
    if args.mode == "ball":
        category_ids = [3]
    elif args.mode == "player":
        category_ids = [4]
    elif args.mode == "all":
        category_ids = [3, 4]
    else:
        print("invalid mode")
        exit(0)
    ratio = args.ratio
    min_area = args.min_area
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "images"))
    os.makedirs(os.path.join(output_path, "images", "train"))
    os.makedirs(os.path.join(output_path, "images", "val"))
    os.makedirs(os.path.join(output_path, "labels"))
    os.makedirs(os.path.join(output_path, "labels", "train"))
    os.makedirs(os.path.join(output_path, "labels", "val"))
    videos = list(glob.iglob("{}/*/*.mp4".format(root), recursive=True))
    annotations = list(glob.iglob("{}/*/*.json".format(root), recursive=True))
    # Most of the times, there will be more videos than annotations. Hence we only consider cases where we have both
    videos = [video.replace(".mp4", "") for video in videos]
    annotations = [anno.replace(".json", "") for anno in annotations]
    paths = list(set(videos) & set(annotations))
    counter = 0
    for idx, path in enumerate(paths):
        print(path)
        video = cv2.VideoCapture("{}.mp4".format(path))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        with open("{}.json".format(path)) as json_file:
            json_data = json.load(json_file)
        images = json_data["images"]
        width = images[0]["width"]
        height = images[0]["height"]
        annotations = json_data["annotations"]
        num_images = len(images)
        num_objects = len(annotations)
        # For each video, verify that video's length is the same as annotated frames
        if num_frames != num_images:
            print("Something is wrong with video {}.mp4 and/or its annotation file".format(path))
            paths.remove(path)

        objects = [[dict_["image_id"] - 1, [(dict_["bbox"][0] + dict_["bbox"][2] / 2) / width,
                                            (dict_["bbox"][1] + dict_["bbox"][3] / 2) / height,
                                            dict_["bbox"][2] / width,
                                            dict_["bbox"][3] / height], dict_["category_id"]] for dict_ in annotations
                   if
                   dict_["category_id"] in category_ids and dict_["attributes"]["occluded"] != "fully_occluded" and
                   dict_["area"] >= min_area]
        if idx < len(paths) * ratio:
            mode = "train"
        else:
            mode = "val"
        frame_id = 0
        while video.isOpened():
            print(idx, frame_id, counter)
            flag, frame = video.read()
            if not flag:
                break
            cv2.imwrite(os.path.join(output_path, "images", mode,
                                     f"{idx}_{frame_id}.jpg"), frame)
            currents_objects = [obj[1] for obj in objects if obj[0] == frame_id]
            currents_categories = [obj[2] for obj in objects if obj[0] == frame_id]
            with open(os.path.join(output_path, "labels", mode,
                                   f"{idx}_{frame_id}.txt"), "w") as f:
                for obj, cat in zip(currents_objects, currents_categories):
                    if len(category_ids) == 1:
                        f.write("0 {:06f} {:06f} {:06f} {:06f}".format(*obj))
                    else:
                        if cat == 4:  # player
                            f.write("0 {:06f} {:06f} {:06f} {:06f}".format(*obj))
                        else:
                            f.write("1 {:06f} {:06f} {:06f} {:06f}".format(*obj))
                    f.write("\n")
            counter += 1
            frame_id += 1
