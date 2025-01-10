import os
from ultralytics import YOLO
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def validate_labels(label_dir, num_classes):
    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            class_idx = int(line.split()[0])
            if class_idx >= num_classes:
                print(f"Invalid class index {class_idx} in {label_file}")

def train_yolo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "gun_dataset", "gun_data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} does not exist. Make sure the dataset preparation step completed successfully.")
        return
    
    validate_labels(os.path.join(base_dir, "gun_dataset", "train/labels"), num_classes=3)  # Adjust `num_classes`
    validate_labels(os.path.join(base_dir, "gun_dataset", "valid/labels"), num_classes=3)  # Adjust `num_classes`
    
    try:
        print("Starting YOLOv8 model training...")
        model = YOLO('yolov8n.pt').to('cuda')
        model.train(
            data=yaml_path,
            epochs=100,
            batch=32,
            imgsz=416,
            name="gun_training",
            patience=30
        )
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    train_yolo()