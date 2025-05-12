import os
import yaml
from collections import defaultdict

# Load class names from YAML
with open("train.yaml", "r") as stream:
    data_yaml = yaml.safe_load(stream)
    class_names = data_yaml["names"]

base_dir = "train"
label_dirs = [f"{base_dir}/labels/train", f"{base_dir}/labels/val", f"{base_dir}/labels/test"]
image_dirs = [f"{base_dir}/images/train", f"{base_dir}/images/val", f"{base_dir}/images/test"]

def count_files(path):
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

def get_counts_by_dir(dirs):
    return [count_files(d) for d in dirs]

# Get file counts
image_counts = get_counts_by_dir(image_dirs)
label_counts = get_counts_by_dir(label_dirs)

# Count images per class
class_image_counts = defaultdict(set)

for dir_path in label_dirs:
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(dir_path, filename)
            with open(filepath, "r") as f:
                classes_in_file = set()
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    class_id = int(parts[0])
                    classes_in_file.add(class_id)
                for class_id in classes_in_file:
                    class_image_counts[class_id].add(filepath)

# Write to dataset_info.txt
with open("dataset_info.txt", "w") as f:
    f.write("YOLO Dataset Information\n")
    f.write("========================\n\n")

    f.write("Folder Structure:\n-----------------\n")
    f.write("train/\n")
    for dir_name, count in zip(
        ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"],
        image_counts + label_counts
    ):
        f.write(f"├── {dir_name:<13} --> {count} files\n")
    
    f.write("\nTotal Image Files:\n------------------\n")
    f.write(f"Train:     {image_counts[0]}\n")
    f.write(f"Validation:{image_counts[1]}\n")
    f.write(f"Test:      {image_counts[2]}\n")
    f.write(f"Total:     {sum(image_counts)}\n\n")

    f.write("Total Label Files:\n------------------\n")
    f.write(f"Train:     {label_counts[0]}\n")
    f.write(f"Validation:{label_counts[1]}\n")
    f.write(f"Test:      {label_counts[2]}\n")
    f.write(f"Total:     {sum(label_counts)}\n\n")

    f.write("Classes:\n--------\n")
    f.write(f"Total number of classes: {len(class_names)}\n\n")
    f.write("Class Names:\n")
    for name in class_names:
        f.write(f"- {name}\n")

    f.write("\nLabel Format:\n-------------\n")
    f.write("Each label file follows YOLO format:\n<class_id> <x_center> <y_center> <width> <height>\n")
    f.write("(All values normalized to [0, 1])\n\n")

    f.write("Class Distribution (by image count):\n------------------------------------\n")
    for class_id, name in enumerate(class_names):
        count = len(class_image_counts[class_id])
        f.write(f"{name:<13}: {count} images\n")
