import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import warnings

EFD_ROOT = '/mnt/d/Seoyeon/EFD-denoised2'
DEST_ROOT = '/mnt/d/Seoyeon/EFD-denoised2-heatmap'
PART_NAMES = {0: "body", 1: "head", 2: "torso", 7: "dress", 3: "l arm", 4: "r arm", 5: "l leg", 6: "r leg"}

def calculate_correlation(matrix_128x8: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = np.corrcoef(matrix_128x8.T)

    # 0으로 나뉘어 발생한 NaN 값을 0으로 채워줌
    corr = np.nan_to_num(corr, nan=0.0)
    return corr

def make_data(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm_3d = np.linalg.norm(series, axis=-1)
    total_x = np.sum(norm_3d, axis=-1)
    corr_total = calculate_correlation(total_x)
    first_x = norm_3d[:, :, 0]
    corr_first = calculate_correlation(first_x)
    return (corr_total, corr_first)

def make_coeffdata(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = series[:, :, 0, :]
    x_cos = calculate_correlation(x[:, :, 0])
    x_sin = calculate_correlation(x[:, :, 1])
    y_cos = calculate_correlation(x[:, :, 2])
    y_sin = calculate_correlation(x[:, :, 3])
    return (x_cos, x_sin, y_cos, y_sin)

def save_heatmap(data: tuple, save_path: str, labels: list, titles: list):
    data_len = len(titles)
    col = int(np.sqrt(data_len - 1)) + 1
    row = (data_len - 1) // col + 1

    diag_mask = np.eye(8, dtype=bool)

    fig, axes = plt.subplots(row, col, figsize=(6*col, 5*row), dpi=300)
    axes_flat = np.atleast_1d(axes).flatten()

    for i in range(data_len):
        ax = axes_flat[i]
        sns.heatmap(data[i], mask=diag_mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    vmin=-1.0, vmax=1.0, xticklabels=labels, yticklabels=labels, 
                    ax=ax, square=True)
        ax.set_title(titles[i], fontsize=12, pad=10)

    for j in range(data_len, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close

    return True

def main():
    corr_list = []

    for name in os.listdir(EFD_ROOT): 
        name_path = os.path.join(EFD_ROOT, name) 
        if not os.path.isdir(name_path): continue 
        for iter in os.listdir(name_path): 
            iter_path = os.path.join(name_path, iter) 
            if not os.path.isdir(iter_path): continue 
            for arr in os.listdir(iter_path): 
                arr_path = os.path.join(iter_path, arr) 
                if not arr_path.endswith(".npy"): continue 
                out_path = os.path.join(DEST_ROOT, name, iter) 
                os.makedirs(out_path, exist_ok=True) 
                contours = np.load(arr_path) # shape: [128, 8, 30, 4] 

                labels = [PART_NAMES[i] for i in range(8)]

                radius = make_data(contours) # shape: [128, 8]
                path = os.path.join(out_path, f"radius-{arr}.png")
                titles = ['sum of radius', 'first radius']
                check1 = save_heatmap(radius, path, labels, titles)
                
                coefficients = make_coeffdata(contours) # shape: [128, 8]
                path = os.path.join(out_path, f"coefficients-{arr}.png")
                titles = ['x cosine', 'x sine', 'y cosine', 'y sine']
                check2 = save_heatmap(coefficients, path, labels, titles)

                if check1 & check2:
                    corr_list.append(list(radius+coefficients))
                    print(f"[SAVED] {name} / {iter} / {arr}")

    # corr_list: list of list of np.ndarray (shape: [8, 8])
    corr_stack = np.stack(corr_list, axis=-1) 
    total = np.nanmean(corr_stack[0], axis=-1)
    first = np.nanmean(corr_stack[1], axis=-1)
    x_cos = np.nanmean(corr_stack[2], axis=-1)
    x_sin = np.nanmean(corr_stack[3], axis=-1)
    y_cos = np.nanmean(corr_stack[4], axis=-1)
    y_sin = np.nanmean(corr_stack[5], axis=-1)

    labels = [PART_NAMES[i] for i in range(8)]
    titles = ['sum of radius', 'first radius']
    path = os.path.join(DEST_ROOT, "radius_heatmap.png")
    save_heatmap((total, first), path, labels, titles)

    titles = ['x cosine', 'x sine', 'y cosine', 'y sine']
    path = os.path.join(DEST_ROOT, "coefficients_heatmap.png")
    save_heatmap((x_cos, x_sin, y_cos, y_sin), path, labels, titles)

    print("[SAVED] FINAL HEATMAPS: radius_heatmap.png, coefficients_heatmap.png")

if __name__ == "__main__":
    main()