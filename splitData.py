import os
import random
import shutil

#paths and split ratio
input_folder = "./Data"    
output_folder = "./Split_Data"
split_ratio = 0.9

#Define destination folders
train_folder = os.path.join(output_folder, "train")
validate_folder = os.path.join(output_folder, "val")

#create folders
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(validate_folder):
    os.makedirs(validate_folder)

#naming counter
num = 0

#Get all subfolders (each representing a class)
subfolders = [file.name for file in os.scandir(input_folder) if file.is_dir()]

#Iterate over each class
for subfolder in subfolders:
    subfolder_path = os.path.join(input_folder, subfolder)
    train_subfolder_path = os.path.join(train_folder, subfolder)
    validate_subfolder_path = os.path.join(validate_folder, subfolder)

    os.makedirs(train_subfolder_path, exist_ok=True)
    os.makedirs(validate_subfolder_path, exist_ok=True)

    #Get all images in class folder
    images = [file.name for file in os.scandir(subfolder_path) if file.is_file()]
    num_images = len(images)
    num_validate = int(num_images * (1 - split_ratio))

    # Randomly select images for validation
    validate_images = set(random.sample(images, num_validate))

    #Copy images into subfolders
    for image in images:
        source_path = os.path.join(subfolder_path, image)
        name = f"{num}.png"

        if os.path.exists(source_path):
            if image in validate_images:
                destination_path = os.path.join(validate_subfolder_path, name)
            else:
                destination_path = os.path.join(train_subfolder_path, name)

            shutil.copyfile(source_path, destination_path)
        else:
            print(f"File not found: {source_path}")

        num += 1