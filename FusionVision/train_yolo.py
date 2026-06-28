from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="test_data/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,          
    device="cpu",      
    workers=4,
    patience=10,       
    project="runs/train",
    name="waste_yolov8n",
    exist_ok=True,
    pretrained=True,   
    verbose=True,
)

print("\n── Treino concluído! ──")
print(f"Melhor modelo salvo em: {results.save_dir}/weights/best.pt")
