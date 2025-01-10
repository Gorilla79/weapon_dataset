# 모델의 위치를 정확히 기입 후 사용!!
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"상위 폴더\runs\detect\gun_training\weights\best.pt")

# Get the class names and the number of classes
class_names = model.names
print(f"Number of classes: {len(class_names)}")
print("Classes:", class_names)
