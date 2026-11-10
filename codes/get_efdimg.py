import os
import numpy as np
import matplotlib.pyplot as plt

EFD_ROOT = "/mnt/d/Seoyeon/EFD-denoised2"
DEST_ROOT = "/mnt/d/Seoyeon/EFD-denoised2"
CLASS_NAMES = ["full_body", "head", "torso", "left_arm", "right_arm", "left_leg", "right_leg", "dress"] 

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

def main(): 
    for name in os.listdir(EFD_ROOT): 
        #name = 'CHS-03' 
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
                contours = np.load(arr_path) # shape: [f, 7, 20, 4] 
                x, y, z = cal_xyz(contours) 
                
                # 그래프 그리기 
                fig = plt.figure(figsize=(9, 6)) 
                ax = fig.add_subplot(111, projection="3d") 
                # part별로 색상 다르게 
                for part in range(8): 
                    mask = (y % 8) == part 
                    ax.plot3D(x[mask], y[mask], z[mask], alpha=0.7, label=f"Part {CLASS_NAMES[part]}") 
                ax.set_xlabel("Frame") 
                ax.set_ylabel("Part") 
                ax.set_zlabel("Equivalent Radius") 
                ax.set_yticks(np.arange(8)) 
                ax.legend() 
                plt.tight_layout() 
                save_name = os.path.splitext(arr)[0]
                save_path = os.path.join(out_path, save_name) 
                plt.savefig( save_path, dpi=200, bbox_inches="tight" ) 
                plt.close(fig) 
                print(f"Saved: {save_path}") 
                    
if __name__ == "__main__": main()