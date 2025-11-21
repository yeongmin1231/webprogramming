import os
import re
import csv
import json
import time
import glob
import ast
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

# ===================== 사용자 설정(필요 시 수정) =====================
DATASET_ROOT = r"C:\label\dataset"  # train/val 아래에 images, labels가 있는 루트
SPLIT = "val"                       # "train" 또는 "val"
IMG_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
LBL_DIR = os.path.join(DATASET_ROOT, SPLIT, "labels")
DATA_YAML = os.path.join(DATASET_ROOT, "data.yaml")

OUTPUT_DIR = os.path.join(DATASET_ROOT, "reviews")
RESULT_CSV = os.path.join(OUTPUT_DIR, f"review_{SPLIT}.csv")
STATE_JSON = os.path.join(OUTPUT_DIR, f"review_{SPLIT}_state.json")

WINDOW_NAME = "YOLO Label Review"
BOX_THICKNESS = 2  # 초기 바운딩박스 두께
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ================================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_names_from_yaml(yaml_path):
    """
    data.yaml에서 names 리스트만 가볍게 파싱 (PyYAML 없이 동작)
    names: ["apple", "mango", "pineapple"] 형태를 가정.
    """
    if not os.path.isfile(yaml_path):
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            text = f.read()
        # names: [...] 줄을 찾고, [...] 부분을 literal_eval
        m = re.search(r"names\s*:\s*(\[.*?\])", text, flags=re.S)
        if not m:
            return None
        return list(ast.literal_eval(m.group(1)))
    except Exception:
        return None

def read_yolo_labels(label_path, img_w, img_h):
    """
    YOLO txt -> 픽셀 좌표 박스 리스트 반환
    각 항목: (class_id, (x1, y1, x2, y2))
    """
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1 = int((xc - w/2) * img_w)
                y1 = int((yc - h/2) * img_h)
                x2 = int((xc + w/2) * img_w)
                y2 = int((yc + h/2) * img_h)
                # 클리핑
                x1 = max(0, min(x1, img_w-1))
                x2 = max(0, min(x2, img_w-1))
                y1 = max(0, min(y1, img_h-1))
                y2 = max(0, min(y2, img_h-1))
                boxes.append((cls, (x1, y1, x2, y2)))
            except Exception:
                continue
    return boxes

def color_for_class(cid):
    # class id에 따른 색상 (BGR)
    rng = np.random.RandomState(cid + 12345)
    return tuple(int(x) for x in rng.randint(60, 255, size=3))

def draw_overlay(img, text_lines, thickness):
    y = 22
    for t in text_lines:
        cv2.putText(img, t, (10, y), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    hint = f"[Keys] O=OK / X=Bad / B=Back / C=Comment / H=Help / +/-=Thickness({thickness}) / Q=Quit+Save"
    cv2.putText(img, hint, (10, img.shape[0]-12), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, hint, (10, img.shape[0]-12), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def load_state():
    if os.path.isfile(STATE_JSON):
        try:
            with open(STATE_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state):
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def append_csv_row(row):
    file_exists = os.path.isfile(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "split", "image_path", "label_path",
            "decision", "num_boxes", "classes", "comment"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    ensure_dirs()
    class_names = load_names_from_yaml(DATA_YAML) or []
    name_map = defaultdict(lambda: "cls?")
    for i, n in enumerate(class_names):
        name_map[i] = n

    # 이미지 리스트 수집
    img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    img_files = []
    for pat in img_exts:
        img_files.extend(glob.glob(os.path.join(IMG_DIR, pat)))
    img_files.sort()
    if not img_files:
        print(f"[!] 이미지가 없습니다: {IMG_DIR}")
        return

    # 진행 상태 복구
    state = load_state()
    idx = int(state.get("index", 0))
    thickness = int(state.get("thickness", BOX_THICKNESS))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW_NAME, 1280, 800)

    help_visible = False
    comment_memory = ""  # 최근 코멘트 기본값

    while 0 <= idx < len(img_files):
        img_path = img_files[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, base + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue
        h, w = img.shape[:2]
        boxes = read_yolo_labels(lbl_path, w, h)

        # 시각화용 복사본
        vis = img.copy()

        # 박스 그리기
        class_counter = defaultdict(int)
        for cls, (x1, y1, x2, y2) in boxes:
            color = color_for_class(cls)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            label = f"{name_map[cls]}({cls})"
            cv2.putText(vis, label, (x1, max(y1-6, 10)), FONT, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (x1, max(y1-6, 10)), FONT, 0.6, color, 2, cv2.LINE_AA)
            class_counter[name_map[cls]] += 1

        info = [
            f"[{SPLIT}] {idx+1}/{len(img_files)}",
            f"Image: {os.path.basename(img_path)}",
            f"Label: {'Yes' if os.path.isfile(lbl_path) else 'No'} | Boxes: {len(boxes)}",
            "Class count: " + (", ".join([f"{k}:{v}" for k,v in class_counter.items()]) if class_counter else "None"),
        ]

        if help_visible:
            info += [
                "O=(OK), X=(Bad), B=(Back), C=(Comment), H=(Help),",
                "+/-=(Thick), Q=(Quit)",
            ]
        draw_overlay(vis, info, thickness)

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q')):
            # 상태 저장 후 종료
            state["index"] = idx
            state["thickness"] = thickness
            save_state(state)
            break

        elif key in (ord('h'), ord('H')):
            help_visible = not help_visible
            continue

        elif key in (ord('+'), ord('=')):
            thickness = min(10, thickness + 1)
            continue

        elif key in (ord('-'), ord('_')):
            thickness = max(1, thickness - 1)
            continue

        elif key in (ord('b'), ord('B')):
            idx = max(0, idx - 1)
            continue

        elif key in (ord('c'), ord('C')):
            # 콘솔 입력으로 코멘트 받기
            cv2.destroyWindow(WINDOW_NAME)
            try:
                comment_memory = input("코멘트를 입력하세요 (Enter=유지 / 공백=없음): ").strip() or comment_memory
            except Exception:
                pass
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
            cv2.resizeWindow(WINDOW_NAME, 1280, 800)
            continue

        elif key in (ord('o'), ord('O'), ord('x'), ord('X')):
            decision = 'OK' if key in (ord('o'), ord('O')) else 'BAD'
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "split": SPLIT,
                "image_path": img_path,
                "label_path": lbl_path if os.path.isfile(lbl_path) else "",
                "decision": decision,
                "num_boxes": len(boxes),
                "classes": ";".join([f"{name_map[c]}" for c, _ in boxes]) if boxes else "",
                "comment": comment_memory,
            }
            append_csv_row(row)
            
            # ✅ 이미지가 바뀌면 코멘트 초기화
            comment_memory = ""  
            
            idx += 1
            state["index"] = idx
            state["thickness"] = thickness
            save_state(state)
            continue


        else:
            # 정의되지 않은 키 → 다음으로 넘어가진 않음
            continue

    cv2.destroyAllWindows()
    print(f"\n[완료] 검수 결과 CSV: {RESULT_CSV}")
    print(f"[저장] 진행 상태 JSON: {STATE_JSON}")

if __name__ == "__main__":
    main()
