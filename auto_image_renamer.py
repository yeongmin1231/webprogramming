import os

# ----------------------------
# 1. 설정 부분
# ----------------------------
# 기본 경로 (다운로드한 이미지 폴더가 들어 있는 곳)
BASE_DIR = r"c:\label\raw_images"   # ⚠️ 여기에 실제 이미지가 있는 상위 폴더 경로 입력
# 예: raw_images/apple, raw_images/mango, raw_images/pineapple

# 클래스 이름 목록 (폴더 이름과 같게)
CLASSES = ["apple", "mango", "pineapple"]

# ----------------------------
# 2. 자동 이름 변경 함수
# ----------------------------
def rename_images():
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
            new_name = f"{cname}{i}{ext}"           # apple1.jpg, mango3.png ...
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)

        print(f"[+] {cname} 폴더 이름 정리 완료!")

# ----------------------------
# 3. 실행부
# ----------------------------
if __name__ == "__main__":
    rename_images()
    print("\n✅ 모든 이미지 이름 정리 완료!")
