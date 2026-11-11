import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")                     # save figures without a display (servers / WSL)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  registers the '3d' projection (matplotlib < 3.2)

from .EFD import extract_part_contours, get_fourier, body_height


def efd_dirname(efd_cfgs):
    """EFD{n} for pyefd-normalized coefficients, EFD{n}-unnorm otherwise (kept apart on purpose)."""
    n = len(efd_cfgs["parts"])
    return f"EFD{n}" if efd_cfgs.get("normalize", False) else f"EFD{n}-unnorm"
from .KillNoise import find_best_window, make_label, denoising


class RUNEFD:

    def __init__(self, path_cfgs, efd_cfgs):
        self.INPUT_PATH = Path(path_cfgs["input_path"])
        self.FOLDER = efd_cfgs["folder"]  # ['name', 'seq', 'cam']
        self.HARMONICS = efd_cfgs["harmonics_num"]
        self.PARTS = efd_cfgs["parts"]
        self.length = len(self.PARTS)
        self.NORMALIZE = efd_cfgs.get("normalize", False)
        self.ALIGN_START = efd_cfgs.get("align_start_point", True)
        self.SCALE_HEIGHT = efd_cfgs.get("scale_to_height", True)
        self.SKIP_EXISTING = efd_cfgs.get("skip_existing", False)

        self.OUTPUT_PATH = Path(path_cfgs["output_path"]) / efd_dirname(efd_cfgs)
        self.IMAGE_EXTENSIONS = {
            ".png", ".jpg", ".jpeg", ".bmp",
            ".webp", ".tif", ".tiff", ".gif"
        }

    def frame2efd(self, folder_path: Path) -> np.ndarray:
        efd_list = []
        initial_arr = np.zeros((self.length, self.HARMONICS, 4), dtype=np.float32)

        for frame_path in sorted(folder_path.iterdir()):
            if not frame_path.is_file(): continue
            if frame_path.suffix.lower() not in self.IMAGE_EXTENSIONS: continue

            parsing = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if parsing is None: continue

            contours = extract_part_contours(parsing, self)
            height = body_height(contours)
            efd_arr = initial_arr.copy()
            for i, (part, contour) in enumerate(contours.items()):
                if i >= len(efd_arr): break
                if contour is None: continue

                efd = get_fourier(contour, self, height)
                if efd is None: continue

                efd_arr[i] = efd
            efd_list.append(efd_arr)

        if not efd_list:
            return np.empty((0, self.length, self.HARMONICS, 4), dtype=np.float32)

        return np.stack(efd_list, axis=0)

    def run(self):
        if not self.INPUT_PATH.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.INPUT_PATH}")
        
        print("========================[EFD PROCESS STARTED]========================")
        current_dirs = [self.INPUT_PATH]
        cnt = 0
        skip = 0
        exist = 0
        frame_sum = 0
 
        for level_idx, level in enumerate(self.FOLDER):
            next_dirs = []

            for parent_dir in current_dirs:
                for child_dir in sorted(parent_dir.iterdir()):
                    if not child_dir.is_dir():
                        continue

                    next_dirs.append(child_dir)
                    if level_idx == len(self.FOLDER) - 1:
                        relative_path = child_dir.relative_to(self.INPUT_PATH)
                        output_path = (self.OUTPUT_PATH / relative_path).with_suffix(".pkl")
                        if self.SKIP_EXISTING and output_path.exists():
                            exist += 1
                            continue

                        efd_npy = self.frame2efd(child_dir)

                        if efd_npy.shape[0] == 0:
                            print(f"[Skipped] Invalid img data: {child_dir}")
                            skip += 1
                            continue

                        frame_sum += efd_npy.shape[0]
                        cnt += 1

                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        tmp_path = output_path.with_suffix(".pkl.tmp")   # write then rename: no half-written .pkl if stopped
                        with open(tmp_path, "wb") as f:
                            pickle.dump(efd_npy, f, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(tmp_path, output_path)
                        
                        if cnt % 500 == 0:
                            mean_frame = frame_sum / cnt
                            print(f"[{cnt} data saved] mean shape: ({mean_frame:.4f}, 8, {self.HARMONICS}, 4)")

            current_dirs = next_dirs

        print(f"[Finished] {cnt} raw EFD data saved, {skip} data skipped, {exist} already existed. save path: {self.OUTPUT_PATH}")
        print("=======================[EFD PROCESS FINISHED]========================")

class RUNDENOISE:

    def __init__(self, path_cfgs, efd_cfgs, denoise_cfgs):
        output_root = Path(path_cfgs["output_path"])
        n_parts = len(efd_cfgs['parts'])
        self.INPUT_PATH = output_root / efd_dirname(efd_cfgs)
        self.OUTPUT_PATH = output_root / f"{efd_dirname(efd_cfgs)}-denoised"

        self.target_shape = (denoise_cfgs['slicing_num'], n_parts, efd_cfgs['harmonics_num'], 4)
        self.part_name = list(efd_cfgs['parts'].keys())
        self.denoise_cfgs = denoise_cfgs

    def pipeline(self, tensor: np.ndarray):
        series, corr_score, corr_mask = find_best_window(tensor=tensor, target_shape=self.target_shape, cfgs=self.denoise_cfgs) # (frame, p, h, 4) -> (f, p, h, 4), 자가상관 점수 (p, ), part mask (p, )
        diagnosis = make_label(series, corr_score, corr_mask, target_shape=self.target_shape,
                               part_name=self.part_name, cfgs=self.denoise_cfgs) # trier, periodicity, label, data
        if diagnosis is None:
            return None
        interp_data = denoising(cfgs=self.denoise_cfgs, diagnosis=diagnosis)
        diagnosis.data = interp_data
        return diagnosis 
    
    def run(self):
        print("=====================[DENOISING PROCESS STARTED]=====================")
        cnt = 0
        skip = 0
        frame_sum = 0
        for file_path in self.INPUT_PATH.rglob("*.pkl"):
            relative_path = file_path.relative_to(self.INPUT_PATH)
            output_path = self.OUTPUT_PATH / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            data = np.asarray(data)
            if data.shape[0] == 0:
                print(f"[Skipped] Empty data: {file_path}")
                skip += 1
                continue

            result = self.pipeline(data)
            if result is None:
                print(f"[Skipped] Shape mismatch {data.shape} vs target {self.target_shape}: {file_path}")
                skip += 1
                continue
            frame_sum += result.data.shape[0]

            with open(output_path, "wb") as f:
                pickle.dump(result.data, f)
            cnt += 1
            if cnt % 50 == 0:
                mean_shape = frame_sum / cnt
                print(f"[{cnt} data saved] mean frame length: {mean_shape:.4f}")
        
        print(f"[Finished] {cnt} denoised EFD data saved, {skip} data skipped. save path: {self.OUTPUT_PATH}")
        print("====================[DENOISING PROCESS FINISHED]=====================")
  
class RUNVIS:
    def __init__(self, path_cfgs, vis_cfgs):
        self.INPUT_PATH = Path(path_cfgs["output_path"])
        self.OUTPUT_PATH = Path(vis_cfgs['vis_path'])
        self.PART = vis_cfgs['part']

    @staticmethod
    def cal_xyz(contours): 
        radius = np.sqrt(np.sum(contours**2, axis=-1))

        f, n_part, _, _ = contours.shape 
        x = np.arange(f)[:, None] 
        x = np.broadcast_to(x, (f, n_part)) 
        y = np.arange(n_part)[None, :] 
        y = np.broadcast_to(y, (f, n_part)) 
        z = radius[:, :, 0]

        x = x.ravel() 
        y = y.ravel() 
        z = z.ravel() 
        return x, y, z

    def run(self):
        print("===================[VISUALIZATION PROCESS STARTED]===================")
        cnt = 0
        skip = 0
        for file_path in self.INPUT_PATH.rglob("*.pkl"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            data = np.asarray(data)

            if data.ndim != 4 or data.shape[0] == 0:
                print(f"[Skipped] Invalid data: {file_path}")
                skip += 1
                continue

            relative_path = file_path.relative_to(self.INPUT_PATH)
            output_path = (self.OUTPUT_PATH / relative_path).with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            x, y, z = self.cal_xyz(data) 
            
            fig = plt.figure(figsize=(9, 6)) 
            ax = fig.add_subplot(111, projection="3d") 
            
            n_part = data.shape[1]
            for part in range(n_part):
                mask = y == part
                name = self.PART[part] if part < len(self.PART) else f"part{part}"
                ax.plot3D(x[mask], y[mask], z[mask], alpha=0.7, label=name)
            ax.set_xlabel("Frame") 
            ax.set_ylabel("Part") 
            ax.set_zlabel("Equivalent Radius") 
            ax.set_yticks(np.arange(n_part))
            ax.legend() 
            plt.tight_layout() 

            try:
                plt.savefig(output_path, dpi=200, bbox_inches="tight" ) 
            finally:
                plt.close(fig) 
            
            cnt += 1
            if cnt % 50 == 0:
                print(f"[{cnt} fig saved]")
        print("==================[VISUALIZATION PROCESS FINISHED]===================")