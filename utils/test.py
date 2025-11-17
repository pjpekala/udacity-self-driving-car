import os
import shutil

source_folder = 'C:/Users/kaiva/Desktop/VC/Dataset/Old'
destination_folder = 'C:/Users/kaiva/Desktop/VC/Dataset/New'

os.makedirs(destination_folder, exist_ok=True)


print(f"Scanning folder: {source_folder}")
all_files = os.listdir(source_folder)
print(f"Found {len(all_files)} files")

for filename in all_files:
    if '_jpg.rf.' in filename and filename.endswith('.jpg'):
       
        new_name = filename.split('_jpg.rf.')[0] + '.jpg'
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, new_name)

        
        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)

            