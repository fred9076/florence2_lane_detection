import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import shutil

def transform_curvelanes(curvelanes_path, output_path):
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = []
        for split in ['train', 'valid']:
            print(f"Processing {split} split...")
            future = executor.submit(process_split, curvelanes_path, output_path, split)
            futures.append(future)

        # Handle test split separately
        print("Copying test images...")
        future = executor.submit(copy_test_images, curvelanes_path, output_path)
        futures.append(future)

        for future in as_completed(futures):
            future.result()

def process_split(curvelanes_path, output_path, split):
    annotations = load_curvelanes_annotations(curvelanes_path, split)
    process_annotations(annotations, curvelanes_path, output_path, split)

def load_curvelanes_annotations(curvelanes_path, split):
    annotations = []
    labels_dir = os.path.join(curvelanes_path, split, 'labels')
    images_dir = os.path.join(curvelanes_path, split, 'images')
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.lines.json'):
            image_file = label_file.replace('.lines.json', '.jpg')
            image_path = os.path.join(images_dir, image_file)
            
            annotations.append({
                'label_path': os.path.join(labels_dir, label_file),
                'image_path': image_path,
            })
    
    return annotations

def process_annotations(annotations, curvelanes_path, output_path, split):
    annotations_file = os.path.join(output_path, split, 'annotations.json')
    
    # Check if the file exists, if not create it
    if not os.path.exists(annotations_file):
        open(annotations_file, 'w').close()
    
    with open(annotations_file, 'r+') as f:
        # Read existing content
        content = f.read()
        existing_annotations = [json.loads(line) for line in content.strip().split('\n') if line]
        
        # Move to the end of the file
        f.seek(0, 2)
        
        for idx, ann in tqdm(enumerate(annotations), total=len(annotations)):
            with open(ann['label_path'], 'r') as label_file:
                label_data = json.load(label_file)

            image = cv2.imread(ann['image_path'])
            height, width = image.shape[:2]

            processed_lanes = process_lanes(label_data['Lines'], height, width)
            florence_ann = create_florence_annotation(processed_lanes, ann['image_path'])

            # Check if the annotation already exists
            if not any(existing_ann['image'] == florence_ann['image'] for existing_ann in existing_annotations):
                save_image_and_annotation(image, florence_ann, output_path, split, idx + len(existing_annotations), f)
            
        # Truncate the file to remove any leftover content
        f.truncate()

def process_lanes(lanes, height, width):
    return [[(float(point['x']) / width, float(point['y']) / height) for point in lane] for lane in lanes]

def create_florence_annotation(lanes, image_path):
    prefix = "<OD_LANE>"
    lane_string = "".join([f"lane{''.join([f'<loc_{int(x*1000)}><loc_{int(y*1000)}>' for x, y in lane])}" for lane in lanes])
    
    return {
        "image": os.path.basename(image_path),
        "prefix": prefix,
        "suffix": lane_string
    }

def save_image_and_annotation(image, annotation, output_path, split, idx, f):
    image_filename = f"curvelanes_{idx:06d}.jpg"
    cv2.imwrite(os.path.join(output_path, split, 'images', image_filename), image)
    
    annotation['image'] = image_filename
    json.dump(annotation, f)
    f.write('\n')

def copy_test_images(curvelanes_path, output_path):
    test_images_dir = os.path.join(curvelanes_path, 'test', 'images')
    output_test_dir = os.path.join(output_path, 'test', 'images')
    
    for idx, image_file in enumerate(os.listdir(test_images_dir)):
        if image_file.endswith('.jpg'):
            src_path = os.path.join(test_images_dir, image_file)
            dst_path = os.path.join(output_test_dir, f"curvelanes_{idx:06d}.jpg")
            shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    curvelanes_path = "/florence2lane/lane_datasets/Curvelanes"
    output_path = "/florence2lane/florence2lane_data/Curvelanes"
    transform_curvelanes(curvelanes_path, output_path)