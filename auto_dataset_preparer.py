import os
import shutil

# ----------------------------
# 1. 사용자 설정
# ----------------------------
BASE_DIR = "C:/label/raw_images"      # 다운로드한 이미지 상위 폴더
OUTPUT_DIR = "C:/label/images"        # 모든 이미지가 모일 폴더
LABEL_DIR = "C:/label/yolo_labels"    # 빈 라벨 파일 생성 폴더
CLASSES = ["apple", "mango", "pineapple"]

# ----------------------------
# 2. 이름 정리 + 복사 함수
# ----------------------------
def organize_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)
    count = 0

    for cname in CLASSES:
        folder = os.path.join(BASE_DIR, cname)
        if not os.path.exists(folder):
            print(f"[!] 폴더 없음: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"[INFO] {cname} 폴더 이미지 {len(files)}개 처리 중...")

        for i, filename in enumerate(files, 1):
            old_path = os.path.join(folder, filename)
            ext = os.path.splitext(filename)[1].lower()
            new_name = f"{cname}{i}{ext}"
            new_img_path = os.path.join(OUTPUT_DIR, new_name)

            shutil.copy(old_path, new_img_path)   # 통합 폴더로 복사

            # 빈 라벨 파일 생성
            label_name = f"{os.path.splitext(new_name)[0]}_mask.txt"
            label_path = os.path.join(LABEL_DIR, label_name)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("")  # 나중에 SAM 마스크 정보로 채워짐

            count += 1

    print(f"\n[+] 총 {count}개의 이미지 통합 및 라벨 파일 생성 완료!")
    print(f"[+] 이미지 폴더: {OUTPUT_DIR}")
    print(f"[+] 라벨 폴더: {LABEL_DIR}")

# ----------------------------
# 3. 실행
# ----------------------------
if __name__ == "__main__":
    organize_images()
    print("\n✅ 이미지 정리 + 라벨 준비 완료! 이제 'yolo_label_converter.py' 또는 SAM 자동 라벨링 실행 가능.")
