from ultralytics import YOLO
import cv2, os

model = YOLO("runs/detect/fruit_train_v35/weights/best.pt")

image_dir = "test_images"
output_dir = "detect_results_v2"
os.makedirs(output_dir, exist_ok=True)

results = model.predict(source=image_dir, conf=0.7, save=False, verbose=False)

for i, result in enumerate(results):
    im = result.orig_img.copy()
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    for box, cls, cf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {cf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cv2.putText(im, label, (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    out_path = os.path.join(output_dir, f"detect_{i+1}.jpg")
    cv2.imwrite(out_path, im)
    print(f"[+] {out_path} 저장 완료")

print("\n✅ 탐지 시각화 완료!")
