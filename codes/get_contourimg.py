import cv2
import os
from EFD import get_contour
import numpy as np

D_PATH = "/mnt/c/Users/HyeBinKim/Desktop/Test/frequency_pilot/Parsing example"
TARGET_COLORS = {
    "HEAD":      [(30, 50, 243)],
    "TORSO":     [(31, 112, 242)],
    "Right_ARM": [(228, 215, 150), (244, 208, 26)],
    "Left_ARM":  [(220, 167, 93), (189, 113, 7)],
    "Right_LEG": [(63, 215, 192), (27, 175, 239)],
    "Left_LEG":  [(156, 209, 158), (12, 234, 242)]
}
DRAW_COLORS = {
    "HEAD": (255, 0, 0),        # 파랑
    "TORSO": (0, 255, 0),       # 초록
    "Right_ARM": (0, 0, 255),   # 빨강
    "Left_ARM": (0, 255, 255), # 노랑
    "Right_LEG": (255, 0, 255), # 보라
    "Left_LEG": (255, 255, 0), # 하늘
}

for img_name in os.listdir(D_PATH):
    img_path = os.path.join(D_PATH, img_name)
    if not img_path.endswith(".png"):
        continue

    img = cv2.imread(img_path)      # 컬러로 읽기
    vis = img.copy()

    for part, colors in TARGET_COLORS.items():

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for color in colors:
            mask |= cv2.inRange(
                img,
                np.array(color, dtype=np.uint8),
                np.array(color, dtype=np.uint8)
            )

        contour = get_contour(mask)   # mask에서 contour 추출

        if contour is None:
            continue

        contour_draw = contour.reshape(-1, 1, 2)

        cv2.drawContours(vis, [contour_draw], -1, DRAW_COLORS[part], 2)

    save_path = os.path.join(
        D_PATH,
        f"{os.path.splitext(img_name)[0]}-cont.jpg"
    )
    cv2.imwrite(save_path, vis)