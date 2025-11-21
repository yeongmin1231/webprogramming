import pandas as pd
import matplotlib.pyplot as plt
import os

# ✅ YOLO 학습 결과 파일 경로
csv_path = r"C:\label\runs\detect\fruit_train_v35\results.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {csv_path}")

# ✅ CSV 불러오기
df = pd.read_csv(csv_path)

# YOLOv8 results.csv의 일반 컬럼 예시
# columns = ['epoch', 'train/box_loss', 'train/cls_loss', 'metrics/precision(B)',
#            'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', ...]

# ✅ 필요한 지표만 추출
epochs = df.index + 1
precision = df["metrics/precision(B)"]
recall = df["metrics/recall(B)"]
map50 = df["metrics/mAP50(B)"]
map95 = df["metrics/mAP50-95(B)"]

# ✅ 그래프 그리기
plt.figure(figsize=(10,6))
plt.plot(epochs, precision, label="Precision", linewidth=2)
plt.plot(epochs, recall, label="Recall", linewidth=2)
plt.plot(epochs, map50, label="mAP@50", linewidth=2)
plt.plot(epochs, map95, label="mAP@50-95", linewidth=2)

plt.title("YOLOv8 Training Metrics (fruit_train_v35)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Metric Value", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# ✅ 그래프 저장
output_path = r"C:\label\runs\detect\fruit_train_v35\metrics_plot.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\n📊 학습 성능 그래프가 저장되었습니다: {output_path}")
