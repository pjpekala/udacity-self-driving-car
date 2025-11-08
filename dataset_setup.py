import os
import random
import shutil

def create_dataset_split(image_folder, label_folder, dataset_output_folder, x_percent, ratios=(0.70, 0.15, 0.15)):
    

    for subfolder in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(dataset_output_folder, subfolder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_output_folder, subfolder, 'labels'), exist_ok=True)


    all_image_names = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith('.jpg')]


    file_pairs = []
    for name in all_image_names:
        image_path = os.path.join(image_folder, f"{name}.jpg")
        label_path = os.path.join(label_folder, f"{name}.txt") # Assumes labels are .txt
        if os.path.exists(image_path) and os.path.exists(label_path):
            file_pairs.append(name)


    num_to_select = int(len(file_pairs) * (x_percent / 100.0))
    selected_pairs = random.sample(file_pairs, num_to_select)
    print(f"Selected {num_to_select} out of {len(file_pairs)} matching file pairs.")


    random.shuffle(selected_pairs)

    # Calculate split sizes
    total_pairs = len(selected_pairs)
    train_size = int(total_pairs * ratios[0])
    valid_size = int(total_pairs * ratios[1])

    # Perform the split
    train_files = selected_pairs[:train_size]
    valid_files = selected_pairs[train_size:train_size + valid_size]
    test_files = selected_pairs[train_size + valid_size:]

    # Copy files to destination folders
    def copy_files(file_list, dest_path, image_src, label_src):
        for name in file_list:
            shutil.copy(os.path.join(image_src, f"{name}.jpg"), os.path.join(dest_path, 'images'))
            shutil.copy(os.path.join(label_src, f"{name}.txt"), os.path.join(dest_path, 'labels'))

    print("Copying files...")
    copy_files(train_files, os.path.join(dataset_output_folder, 'train'), image_folder, label_folder)
    copy_files(valid_files, os.path.join(dataset_output_folder, 'valid'), image_folder, label_folder)
    copy_files(test_files, os.path.join(dataset_output_folder, 'test'), image_folder, label_folder)
    print("Dataset split complete.")

# --- Example usage ---
if __name__ == "__main__":
    IMAGE_FOLDER = "C:/Users/kaiva/Desktop/GitHub/Udacity Self Driving Car/udacity-self-driving-car/Dataset/image"
    LABEL_FOLDER = "C:/Users/kaiva/Desktop/GitHub/Udacity Self Driving Car/udacity-self-driving-car/Dataset/label"
    DATASET_OUTPUT_FOLDER = "C:/Users/kaiva/Desktop/GitHub/Udacity Self Driving Car/udacity-self-driving-car/Dataset_Random_20%"
    PERCENT_TO_SELECT = 20  # Selects 20% of the matching image/label pairs
    
    # Check if the folders exist
    if not os.path.isdir(IMAGE_FOLDER):
        print(f"Error: Image folder not found at {IMAGE_FOLDER}")
    elif not os.path.isdir(LABEL_FOLDER):
        print(f"Error: Label folder not found at {LABEL_FOLDER}")
    else:
        create_dataset_split(
            image_folder=IMAGE_FOLDER,
            label_folder=LABEL_FOLDER,
            dataset_output_folder=DATASET_OUTPUT_FOLDER,
            x_percent=PERCENT_TO_SELECT
        )

