from ultralytics import YOLO

# TODO Make this more user-friendly as a CLI script

# Initialize the YOLO model
model = YOLO('yolov8n.pt')

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data='./tmp/random/random.yaml', epochs=50, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)