import cv2
import numpy as np
import os
from pyefd import elliptic_fourier_descriptors
import pandas as pd

COLORS = {
        'head': (0, 0, 255),
        'torso': (0, 255, 0),
        'left_arm': (255, 0, 0),
        'right_arm': (255, 255, 0),
        'left_leg': (0, 255, 255),
        'right_leg': (255, 0, 255),
        'dress': (128, 128, 128),
        'background': (255, 255, 255)}

N = 30

def extract_part_contours(parsing_result, runner):
    contours_dict = {}
    for class_name, class_id in runner.PARTS.items():
        if class_name == 'background':
            mask = np.uint8(parsing_result != class_id) * 255
        else:
            mask = np.uint8(np.isin(parsing_result, class_id)) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(mask,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
        else: contour = None
        contours_dict[class_name] = contour

    return contours_dict

def align_start_point(coeffs):
    """Phase (start-point) normalization only: the first step of pyefd.normalize_efd.
    Removes the dependency on where cv2.findContours starts the contour, keeps rotation and size."""
    theta_1 = 0.5 * np.arctan2(
        2 * (coeffs[0, 0] * coeffs[0, 1] + coeffs[0, 2] * coeffs[0, 3]),
        coeffs[0, 0] ** 2 - coeffs[0, 1] ** 2 + coeffs[0, 2] ** 2 - coeffs[0, 3] ** 2)
    n = np.arange(1, coeffs.shape[0] + 1)
    cos, sin = np.cos(n * theta_1), np.sin(n * theta_1)
    rot = np.stack([np.stack([cos, -sin], 1), np.stack([sin, cos], 1)], 1)    # [N, 2, 2]
    return np.matmul(coeffs.reshape(-1, 2, 2), rot).reshape(-1, 4)


def get_fourier(contour, runner, body_height=None):
    """
    runner.NORMALIZE      True : pyefd normalization (rotation / size / start-point invariant)
                          False: raw coefficients, optionally
    runner.ALIGN_START    -> start-point (phase) alignment only
    runner.SCALE_HEIGHT   -> divide by the per-frame body height (removes camera distance)
    """
    n = runner.HARMONICS
    contour = contour.reshape(-1, 2)
    if len(contour) < 3:
        return None
    if runner.NORMALIZE:
        return elliptic_fourier_descriptors(contour, order=n, normalize=True)
    efd = elliptic_fourier_descriptors(contour, order=n, normalize=False)
    if runner.ALIGN_START:
        efd = align_start_point(efd)
    if runner.SCALE_HEIGHT and body_height:
        efd = efd / body_height
    return efd


def body_height(contours_dict):
    """y-extent (px) of all part contours of a frame = full-body height."""
    pts = [c.reshape(-1, 2) for c in contours_dict.values() if c is not None]
    if not pts:
        return None
    y = np.concatenate(pts)[:, 1]
    h = float(y.max() - y.min())
    return h if h > 0 else None

def save_contour_image(img, contour_dict, save_path):
    result = img.copy()

    overlay = result.copy()

    for class_name, contour in contour_dict.items():
        if contour is None:
            continue

        color = COLORS.get(class_name, (255, 255, 255))
        cv2.drawContours(overlay, [contour], -1, color, thickness=-1)

    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, result, 1-alpha, 0)

    color = COLORS['background']
    cv2.drawContours(result, [contour_dict['background']], -1, color, thickness=1)

    cv2.imwrite(save_path, result)
    return True

'''
def main():
    print("Running EFD ...")

    for name in os.listdir(FRAME_PATH):
        name_path = os.path.join(FRAME_PATH, name)
        if not os.path.isdir(name_path): continue
        for iter in os.listdir(name_path):
            iter_path = os.path.join(name_path, iter)
            if not os.path.isdir(iter_path): continue
            for dir in os.listdir(iter_path):
                dir_path = os.path.join(iter_path, dir, "Pred_parsing_results")
                if not os.path.isdir(dir_path): continue

                out_path = os.path.join(OUT_PATH, name, iter)
                os.makedirs(out_path, exist_ok=True)
                efd_list = [] 

                pic = True
                for frame in sorted(os.listdir(dir_path)):
                    frame_path = os.path.join(dir_path, frame)
                    if not frame_path.endswith(".png"): continue

                    parsing = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

                    contours = extract_part_contours(parsing)
                    if pic:
                        img = cv2.imread(frame_path)
                        save_contour_image(img, contours, os.path.join(out_path, f"{dir}.png"))
                        pic = False

                    initial_arr = np.zeros((8, N, 4), dtype=np.float32)
                    i = 0
                    for part, contour in contours.items():
                        if contour is None:
                            i += 1 
                            continue
                        efd = get_fourier(contour, N)
                        if efd is None:
                            i += 1
                            continue
                        initial_arr[i] = efd
                        i += 1
                    efd_list.append(initial_arr)

                efd_arr = np.stack(efd_list, axis=0)
                np.save(os.path.join(out_path, f"{dir}.npy"), efd_arr)
                print(f"saved: {name}-{iter}-{dir} | {efd_arr.shape}")

if __name__ == "__main__":
    main()
'''