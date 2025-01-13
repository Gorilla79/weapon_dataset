import os
import random
import shutil
from ultralytics import YOLO

# Allow OpenMP duplication
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define paths to datasets and models
datasets = {
    "hammer": r"C:\Users\415-28\Desktop\dongsup\hammer_dataset\valid",
    "knife": r"C:\Users\415-28\Desktop\dongsup\knife_dataset\valid",
    "gun": r"C:\Users\415-28\Desktop\dongsup\gun_dataset\valid",
    "bat": r"C:\Users\415-28\Desktop\dongsup\bat_dataset\valid",
    "axe": r"C:\Users\415-28\Desktop\dongsup\axe_dataset\valid",
}

models = {
    "hammer": r"C:\Users\415-28\Desktop\dongsup\runs\detect\hammer_training\weights\best.pt",
    "knife": r"C:\Users\415-28\Desktop\dongsup\runs\detect\knife_training\weights\best.pt",
    "gun": r"C:\Users\415-28\Desktop\dongsup\runs\detect\gun_training\weights\best.pt",
    "bat": r"C:\Users\415-28\Desktop\dongsup\runs\detect\bat_training\weights\best.pt",
    "axe": r"C:\Users\415-28\Desktop\dongsup\runs\detect\axe_training\weights\best.pt",
}

# Random sampling and evaluation
def evaluate_random_samples(dataset_path, model_path, num_samples=50):
    # Get image files
    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")
    image_files = os.listdir(image_dir)
    random_samples = random.sample(image_files, min(num_samples, len(image_files)))






    # Prepare a temporary dataset
    temp_dir = "./temp"
    temp_images_dir = os.path.join(temp_dir, "images")
    temp_labels_dir = os.path.join(temp_dir, "labels")
    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    for sample in random_samples:
        base_name = os.path.splitext(sample)[0]
        image_src = os.path.join(image_dir, sample)
        label_src = os.path.join(label_dir, f"{base_name}.txt")
        image_dst = os.path.join(temp_images_dir, sample)
        label_dst = os.path.join(temp_labels_dir, f"{base_name}.txt")
        shutil.copy(image_src, image_dst)
        if os.path.exists(label_src):  # Check if label file exists
            shutil.copy(label_src, label_dst)

    # Create a YAML file for the temporary dataset
    yaml_path = os.path.join(temp_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(temp_dir)}\n")
        f.write(f"train: {os.path.abspath(temp_images_dir)}\n")
        f.write(f"val: {os.path.abspath(temp_images_dir)}\n")
        f.write(f"names: [\"{os.path.basename(dataset_path)}\"]\n")

    # Evaluate using YOLOv8
    model = YOLO(model_path)
    result = model.val(data=yaml_path)
    print(result)

    # Cleanup temporary directory
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Run evaluation for all datasets
    num_samples_per_dataset = 50
    for class_name, dataset_path in datasets.items():
        print(f"Evaluating {class_name} with {num_samples_per_dataset} random samples...")
        evaluate_random_samples(dataset_path, models[class_name], num_samples_per_dataset)