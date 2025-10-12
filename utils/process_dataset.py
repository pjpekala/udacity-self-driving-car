import json
import os
from collections import defaultdict
import argparse

def process_coco_dataset(coco_file_path, output_file_path):
    """
    Deduplicates, renames, and reorders a COCO dataset.
    This script combines the functionality of deduplicate.py, rename_coco_files.py, and reorder_ids.py.
    """
    
    coco_dir = os.path.dirname(coco_file_path)

    # Step 1: Deduplicate images and annotations
    try:
        with open(coco_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {coco_file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {coco_file_path}.")
        return

    print("Step 1: Finding and removing duplicate images...")
    
    image_groups = defaultdict(list)
    for image in data['images']:
        prefix = image['file_name'].split('.rf.')[0]
        image_groups[prefix].append(image)

    images_to_keep = []
    images_to_remove = []
    
    for prefix, group in image_groups.items():
        images_to_keep.append(group[0])
        images_to_remove.extend(group[1:])

    image_ids_to_remove = {image['id'] for image in images_to_remove}
    filenames_to_remove = {image['file_name'] for image in images_to_remove}

    print(f"Found {len(filenames_to_remove)} duplicate images to remove.")

    if 'annotations' in data:
        data['annotations'] = [
            ann for ann in data['annotations'] if ann['image_id'] not in image_ids_to_remove
        ]

    data['images'] = images_to_keep

    removed_count = 0
    for filename in filenames_to_remove:
        try:
            image_path = os.path.join(coco_dir, filename)
            os.remove(image_path)
            removed_count += 1
        except FileNotFoundError:
            print(f"Warning: {image_path} not found, skipping removal.")
        except OSError as e:
            print(f"Error removing {image_path}: {e}")

    print(f"Successfully removed {removed_count} duplicate image files.")
    print("Step 1: Deduplication complete.")
    print("-" * 20)

    # Step 2: Rename image files and update filenames in annotations
    print("Step 2: Renaming image files...")
    renamed_files_count = 0
    for image in data['images']:
        old_filename = image['file_name']
        if '_jpg.rf.' in old_filename:
            base_name = old_filename.split('_jpg.rf.')[0]
            new_filename = base_name + '.jpg'
            
            old_path = os.path.join(coco_dir, old_filename)
            new_path = os.path.join(coco_dir, new_filename)

            try:
                os.rename(old_path, new_path)
                image['file_name'] = new_filename
                renamed_files_count += 1
            except FileNotFoundError:
                print(f"Warning: {old_path} not found, skipping rename.")
            except OSError as e:
                print(f"Error renaming {old_path} to {new_path}: {e}")
    
    print(f"Successfully renamed {renamed_files_count} image files.")
    print("Step 2: Renaming complete.")
    print("-" * 20)

    # Step 3: Reorder IDs for images and annotations
    print("Step 3: Reordering IDs...")
    image_id_map = {}
    for i, image in enumerate(data['images']):
        image_id_map[image['id']] = i
        image['id'] = i

    if 'annotations' in data:
        # Update image_id in annotations
        for annotation in data['annotations']:
            if annotation['image_id'] in image_id_map:
                annotation['image_id'] = image_id_map[annotation['image_id']]
        
        # Update annotation ids
        for i, annotation in enumerate(data['annotations']):
            annotation['id'] = i
    
    print("Successfully reordered IDs.")
    print("Step 3: Reordering complete.")
    print("-" * 20)

    # Step 4: Write the updated annotations to a new file
    print(f"Step 4: Writing processed data to {output_file_path}...")
    try:
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully created processed annotations file at {output_file_path}.")
    except IOError as e:
        print(f"Error writing to {output_file_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process a COCO dataset by deduplicating, renaming, and reordering IDs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_file', 
        help="Path to the input COCO annotations file (e.g., self_driving_car_coco/export/_annotations.coco.json)."
    )
    parser.add_argument(
        'output_file', 
        help="Path to save the processed COCO annotations file (e.g., self_driving_car_coco/export/_annotations.processed.coco.json)."
    )
    
    args = parser.parse_args()
    
    process_coco_dataset(args.input_file, args.output_file)

# to run:
# python /Users/pjpekala/school_stuff/udacity-self-driving-car/utils/process_dataset.py /Users/pjpekala/school_stuff/udacity-self-driving-car/self_driving_car_coco/export/_annotations.coco.json /Users/pjpekala/school_stuff/udacity-self-driving-car/self_driving_car_coco/export/annotations.processed.json