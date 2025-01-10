from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"C:\Users\415-28\Desktop\dongsup\runs\detect\gun_training\weights\best.pt")

# Get the class names and the number of classes
class_names = model.names
print(f"Number of classes: {len(class_names)}")
print("Classes:", class_names)
