import numpy as np
import pyefd
import matplotlib.pyplot as plt

test_data_path = "/mnt/d/Seoyeon/EFD/HYJ/nm-03/DB.npy"
test_check_path = "/mnt/d/Seoyeon"

with open(test_data_path, 'rb') as f:
    coeffs = np.load(f)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i in range(2):
    for j in range(4):
        p = i*4 + j
        coeff = coeffs[0, p, :, :]  # 각 파트의 EFD 계수 선택

        # 2. 외곽선 복원
        reconstructed_contour = pyefd.reconstruct_contour(coeff, locus=(0, 0), num_points=300)

        # 3. 복원된 외곽선 시각화

        axes[i, j].plot(
            reconstructed_contour[:, 0],
            reconstructed_contour[:, 1],
            color='black',
            linewidth=2
        )
        axes[i, j].set_title(f"Part {p + 1}")
        axes[i, j].axis('equal')
        axes[i, j].grid(True)

plt.title("Reconstructed Contour")
plt.tight_layout()

# 4. 이미지 저장
plt.savefig(
    f"{test_check_path}/reconstructed_contour.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.show()