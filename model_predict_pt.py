from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

yolo = YOLO("yolo11n.pt", task="detect")
results = yolo.predict("bus.jpg")[0]

annotated_img = results.plot()
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.title("YOLO11 Detection Results")
plt.axis("off")
plt.tight_layout()
plt.savefig("results_pytorch.png", bbox_inches="tight", facecolor="white")
