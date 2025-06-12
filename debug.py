import cv2
import numpy as np
import torch
import random
import sys
# sys.path.append(".utils/")  # 상위 디렉토리 추가

# 기존 함수 (old version)
from utils.augmentations import random_perspective_rgb_ir as old_random_perspective

# 새로 재작성한 함수 (new version)
from utils.augmentations import random_perspective_rgb_ir_2 as new_random_perspective


# ───────────────────────────────────────────────────────────────
# 1. Dummy input 생성
# ───────────────────────────────────────────────────────────────
h, w = 480, 640
img_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
img_ir = np.full((h, w, 3), 100, dtype=np.uint8)

# 정사각형 bbox (중앙에 위치)
targets = np.array([[0, 200, 150, 300, 250]], dtype=np.float32)  # cls, x1, y1, x2, y2

# YOLO format으로 변환 → (cls, cx, cy, w, h)
targets_yolo = targets.copy()
targets_yolo[:, 1] = (targets[:, 1] + targets[:, 3]) / 2 / w
targets_yolo[:, 2] = (targets[:, 2] + targets[:, 4]) / 2 / h
targets_yolo[:, 3] = (targets[:, 3] - targets[:, 1]) / w
targets_yolo[:, 4] = (targets[:, 4] - targets[:, 2]) / h

segments = []  # 비어 있음

# YOLO to corner xyxy
def yolo_to_xyxy(targets_yolo, w, h):
    xyxy = targets_yolo.copy()
    xyxy[:, 1] = (targets_yolo[:, 1] - targets_yolo[:, 3] / 2) * w
    xyxy[:, 2] = (targets_yolo[:, 2] - targets_yolo[:, 4] / 2) * h
    xyxy[:, 3] = (targets_yolo[:, 1] + targets_yolo[:, 3] / 2) * w
    xyxy[:, 4] = (targets_yolo[:, 2] + targets_yolo[:, 4] / 2) * h
    return xyxy

# ───────────────────────────────────────────────────────────────
# 2. 동일한 seed로 두 함수 호출
# ───────────────────────────────────────────────────────────────
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

img_rgb_old, img_ir_old, targets_old, _ = old_random_perspective(
    img_rgb.copy(), img_ir.copy(), targets_yolo.copy(), targets_yolo.copy(), segments, segments)

random.seed(random_seed)
np.random.seed(random_seed)

img_rgb_new, img_ir_new, targets_new, _ = new_random_perspective(
    img_rgb.copy(), img_ir.copy(), targets_yolo.copy(), targets_yolo.copy(), segments, segments)

# ───────────────────────────────────────────────────────────────
# 3. 시각적 결과 확인 (바운딩 박스 출력)
# ───────────────────────────────────────────────────────────────
def draw_boxes(img, targets, color=(0, 255, 0)):
    img = img.copy()
    boxes = yolo_to_xyxy(targets, img.shape[1], img.shape[0]).astype(int)
    for b in boxes:
        cv2.rectangle(img, (b[1], b[2]), (b[3], b[4]), color, 2)
    return img

img_old_bb = draw_boxes(img_rgb_old, targets_old, color=(255, 0, 0))
img_new_bb = draw_boxes(img_rgb_new, targets_new, color=(0, 255, 0))

# ───────────────────────────────────────────────────────────────
# 4. 비교 시각화 (좌: 기존, 우: 재작성)
# ───────────────────────────────────────────────────────────────
from matplotlib import pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_old_bb[:, :, ::-1])
plt.title("Old random_perspective")

plt.subplot(1, 2, 2)
plt.imshow(img_new_bb[:, :, ::-1])
plt.title("New random_perspective")

plt.show()

# ───────────────────────────────────────────────────────────────
# 5. 수치 비교 (좌표 차이 확인)
# ───────────────────────────────────────────────────────────────
print("Box difference (L1 norm):", np.abs(targets_old[:, 1:5] - targets_new[:, 1:5]).sum())