from ultralytics import YOLO

yolo = YOLO("yolo11n.pt")

yolo.export(format="coreml", nms=True)
