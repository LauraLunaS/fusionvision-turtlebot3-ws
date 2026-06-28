from ultralytics import YOLO

model = YOLO("runs/detect/runs/train/waste_yolov8n/weights/best.pt")

results = model.train(
    data="test_data/dataset_combined/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device="cpu",
    workers=4,
    patience=15,
    lr0=0.001,     
    lrf=0.001,
    freeze=10,  
    project="runs/train",
    name="waste_yolov8n_combined_v2",
    exist_ok=True,
    verbose=True,
)
