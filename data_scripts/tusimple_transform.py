#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate TuSimple training dataset

Original code from: https://github.com/MaybeShewill-CV/lanenet-lane-detection
Dataset: https://github.com/TuSimple/tusimple-benchmark/issues/3

Changes by Iroh Cao:
1. Added validation set/test set
2. Fixed small errors

Further refactoring for improved structure and readability
"""

import argparse
import glob
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from shapely.geometry import LineString

def parse_args():
    parser = argparse.ArgumentParser(description='Generate TuSimple training dataset')
    parser.add_argument('--src_dir', type=str, required=True, help='Path to unzipped TuSimple dataset')
    parser.add_argument('--dst_dir', type=str, required=True, help='Destination path for processed dataset')
    parser.add_argument('--val', action='store_true', help='Generate validation set')
    return parser.parse_args()

def downsample_lane(lane_x, lane_y, num_points=10):
    line = LineString(zip(lane_x, lane_y))
    total_length = line.length
    distances = [i * total_length / (num_points - 1) for i in range(num_points)]
    points = [line.interpolate(distance) for distance in distances]
    new_x, new_y = zip(*[(point.x, point.y) for point in points])
    new_x = list(new_x)
    new_y = list(new_y)
    new_x[0], new_y[0] = lane_x[0], lane_y[0]
    new_x[-1], new_y[-1] = lane_x[-1], lane_y[-1]
    return new_x, new_y

def process_json_file(json_file_path, src_dir, ori_dst_dir, output_json_file):
    assert Path(json_file_path).exists(), f'{json_file_path} does not exist'

    image_nums = len(os.listdir(ori_dst_dir))
    prefix = "<OD_LANE>"
    class_name = "lane"
    annotations = []
    dataset_name = 'tulane'

    with open(json_file_path, 'r') as file:
        for img_index, line in enumerate(file):
            info_dict = json.loads(line)
            image_path = Path(src_dir) / info_dict['raw_file']
            assert image_path.exists(), f'{image_path} does not exist'

            h_samples = info_dict['h_samples']
            lanes = info_dict['lanes']

            image_name_new = f'{dataset_name}_{img_index:04d}.png'
            src_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            suffix_lines = []
            for lane in lanes:
                lane_points = [(x, y) for x, y in zip(lane, h_samples) if x != -2]
                if len(lane_points) < 2:
                    continue
            

                loc_list_lane = ''.join(f'<loc_{int(x/src_image.shape[1]*1000)}><loc_{int(y/src_image.shape[0]*1000)}>'
                                        for x, y in lane_points)
                suffix_lines.append(f'{class_name}{loc_list_lane}')

            dst_rgb_image_path = Path(ori_dst_dir) / image_name_new
            cv2.imwrite(str(dst_rgb_image_path), src_image)

            json_obj = {
                "image": image_name_new,
                "prefix": prefix,
                "suffix": "".join(suffix_lines)
            }
            annotations.append(json_obj)
            print(f'Processed {image_name_new}')

    with open(output_json_file, 'w') as json_file:
        for annotation in annotations:
            json.dump(annotation, json_file)
            json_file.write('\n')

    print(f"Annotations have been written to {output_json_file}")

def process_tusimple_dataset(src_dir, dst_dir, val_tag):
    src_dir = Path(src_dir)
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    train_folder_path = Path(dst_dir) / 'train'
    val_folder_path = Path(dst_dir) / 'valid'

    train_folder_path.mkdir(exist_ok=True)
    val_folder_path.mkdir(exist_ok=True)

    for json_label_path in src_dir.glob('label*.json'):
        shutil.copy(json_label_path, train_folder_path)

    for json_label_path in src_dir.glob('test*.json'):
        shutil.copy(json_label_path, val_folder_path)

    gt_image_dir = train_folder_path / 'images'
    gt_json_file = train_folder_path / 'annotations.json'

    gt_image_dir.mkdir(exist_ok=True)

    for json_label_path in train_folder_path.glob('label*.json'):
        process_json_file(json_label_path, src_dir, gt_image_dir, gt_json_file)

    if val_tag:
        gt_image_dir_val = val_folder_path / 'images'
        gt_json_file_val = val_folder_path / 'annotations.json'

        gt_image_dir_val.mkdir(exist_ok=True)

        for json_label_path in val_folder_path.glob('test*.json'):
            process_json_file(json_label_path, src_dir, gt_image_dir_val, gt_json_file_val)

def main():
    args = parse_args()
    process_tusimple_dataset(args.src_dir, args.dst_dir, args.val)

if __name__ == '__main__':
    main()