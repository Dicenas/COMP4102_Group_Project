from ultralytics import YOLO

model = YOLO('')

results = model(source=0, show=True, conf=0.4, save=False)