from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.stats import kurtosis
from scipy.signal import  find_peaks
from dataclasses import dataclass
import os
import scipy.ndimage as ndimage


#------------------------------------------------------------------------
# compute_part_series: 텐서 축약 함수 - (h, 4) coeff -> (1) radius 합
# cal_autocorr: 자가상관 계산함수
#------------------------------------------------------------------------
def compute_part_series(tensor: np.ndarray) -> np.ndarray:
    # [F, p, h, 4] → [F, p]
    assert tensor.ndim == 4, f"expected 4D tensor, got shape {tensor.shape}"
    magnitude = np.sqrt(np.sum(tensor ** 2, axis=-1))   # (F, p, H) per-harmonic radius
    series = magnitude.sum(axis=-1)                      # (F, p) summed over harmonics
    return series

def cal_autocorr(x: np.ndarray, dir: int = 0, n_frames: int = 128) -> np.ndarray:
    x_centered = x - np.mean(x, axis=dir, keepdims=True)

    n_fft = 2 ** int(np.ceil(np.log2(2 * n_frames - 1)))
    X_fft = np.fft.rfft(x_centered, n=n_fft, axis=dir)
    power_spectrum = np.abs(X_fft) ** 2
    autocorr = np.fft.irfft(power_spectrum, n=n_fft, axis=dir)

    slice_obj = [slice(None)] * x.ndim
    slice_obj[dir] = slice(0, n_frames)
    autocorr_cropped = autocorr[tuple(slice_obj)]

    lag0_slice = [slice(None)] * x.ndim
    lag0_slice[dir] = slice(0, 1)
    autocorr_norm = autocorr_cropped / (autocorr_cropped[tuple(lag0_slice)] + 1e-8)
    return autocorr_norm

#------------------------------------------------------------------------
# 슬라이싱 함수
# check_periodicity: 입력 tensor의 주기성을 자가상관으로 계산
# fill_with_period: 입력 tensor에서 중앙 주기를 이용해 128frame이 되도록 보간
# find_best_window: 입력 tensor에서 128 frame window를 슬라이싱하며 최적 탐지
#------------------------------------------------------------------------
def check_periodicity(tensor: np.ndarray, window_cfgs, target_frame: int, auto_axis: int = 0) -> tuple[float, int | None, np.ndarray | None, np.ndarray | None]: 
    valid_parts = None
    period_range = window_cfgs['period_range']
    # tensor 형식 통일 (128, parts)
    x = np.asarray(tensor, dtype=float) #자료형 체크
    if x.shape[0] != target_frame: return 0.0, None, None, None
    if x.ndim > 1:
        #대부분 0인 값 제외
        zero_ratio = np.mean(x==0, axis=0)
        valid_parts = zero_ratio < 0.8
        if not valid_parts.any(): return 0.0, None, None, None
        x = x[:, valid_parts]

        x = x - np.mean(x, axis=0, keepdims=True) # 평균 0으로 고정
        x = x / (np.std(x, axis=0, keepdims=True) + 1e-8) # part별 크기 조정
    else:
        x = x[:, None]
        x = x - np.mean(x, axis=0, keepdims=True)

    ac_2d = cal_autocorr(x=x, dir=auto_axis, n_frames=target_frame)

    # 유효한 part만 합산하여 평균
    mask = np.abs(ac_2d) >= 0.2
    ac = np.sum(np.where(mask, ac_2d, 0), axis=1) / (np.sum(mask, axis=1) + 1e-8)

    lo, hi = period_range #주기 범위 지정
    hi = min(hi, target_frame)
    if lo >= hi: return 0.0, None, None, None

    # 구간 내 자가상관 최대값 탐색
    local_ac = ac[lo:hi]
    peak_idx = np.argmax(local_ac)
    score = float(local_ac[peak_idx])
    period = int(peak_idx + lo)
    part_score = ac_2d[period, :]

    if score <= 0.2 or period <= 0 or period >= target_frame:
        return 0.0, None, None, None

    # 최종 사용된 part mask (대부분 값이 0이거나, 자가상관이 0.2이하로 매우 낮은 part에 False)
    if valid_parts is not None:
        final_mask = np.zeros(len(valid_parts), dtype=bool)
        final_score = np.zeros(len(valid_parts), dtype=float)

        final_mask[valid_parts] = mask[period, :]
        final_score[valid_parts] = part_score
    else: 
        final_mask = mask[period]
        final_score = part_score

    return score, period, final_mask, final_score

def fill_with_period(tensor: np.ndarray, window_cfgs, target_frame: int, auto_axis: int = 0) -> np.ndarray:
    frame = tensor.shape[0] # (frame, p)
    if frame >= target_frame:
        return tensor

    _, period, _, _ = check_periodicity(compute_part_series(tensor), window_cfgs, frame, auto_axis)

    #주기를 발견하지 못했을 경우 또는 주기가 없을 경우단순 반복
    if period is None or period <= 0 or frame < period:
        adds = int(np.ceil(target_frame/frame))
        padded = np.tile(tensor, (adds,) + (1,) * (tensor.ndim - 1))
        return padded[:target_frame]
    
    #발견한 경우
    needed = target_frame - frame
    left = frame % period
    count = frame // period
    
    middle = count // 2
    start = period * middle + left
    end = start+period
    pattern = tensor[start:end] #최대한 중앙에 위치한 period를 가져옴

    adds = int(np.ceil(needed/len(pattern)))
    padding = np.tile(pattern, (adds,) + (1,) * (tensor.ndim - 1))[:needed]
    return np.concatenate([tensor, padding], axis=0)

def find_best_window(tensor: np.ndarray, target_shape: tuple, cfgs) -> tuple[np.ndarray, np.ndarray , np.ndarray]:
    # tensor (frame, p, h, 4)
    # radius (frame, p)
    window_cfgs = cfgs['window']
    frame, _, _, _ = tensor.shape
    target_frame = target_shape[0]
    if frame <= target_frame:
        tensor = fill_with_period(tensor, window_cfgs, target_frame, cfgs['autocorr_axis'])
        frame = tensor.shape[0]

    radius = compute_part_series(tensor)

    scores = np.zeros(frame - target_frame + 1)
    part_scores = [None] * len(scores)
    periods = [None] * len(scores)
    masks = [None] * len(scores)

    for i in range(frame - target_frame + 1):
        window = radius[i:i+target_frame]
        scores[i], periods[i], masks[i], part_scores[i] = check_periodicity(window, window_cfgs, target_frame, cfgs['autocorr_axis'])

    # final_tensor (f, p, h, 4)
    final_idx = np.argmax(scores)
    final_score = part_scores[final_idx] #(p, )
    final_mask = masks[final_idx] # (p, )
    if final_mask is None: # 주기를 찾은 window가 없는 경우
        final_score = np.zeros(radius.shape[1], dtype=float)
        final_mask = np.zeros(radius.shape[1], dtype=bool)
    final_tensor = tensor[final_idx:final_idx+target_frame] # (f, p, h, 4)
    # 최종적으로 슬라이싱된 (f, p) tensor와 자가상관 점수
    return final_tensor, final_score, final_mask

#------------------------------------------------------------------------
# 유형 분류 함수
# interpolate_outlier: 입력 tensor의 Cubic Spline으로 outlier 제거
# get_periodicity_mask: 입력 자가상관의 주기성을 판별해 마스킹 제공 (period True)
# check_drop_mask: 0에 가까운 drop값의 비율&연속성을 판별해 마스킹 제공 (drop True)
# make_label: 유형 분류
#------------------------------------------------------------------------
@dataclass
class Diagnosis:
    periodicity: np.ndarray # (f, p, h, 4)
    label: np.ndarray  # zero, drop, clean, noise | np.array (p, h, 4)
    data: np.ndarray # (f, p, h, 4)

def interpolate_outlier(x: np.ndarray, threshold: float = 3.0, pulse_threshold: float = 2.0, 
                        n_frames: int = 128, n_harmonics: int = 32) -> np.ndarray:
    # x  (f, h, 4)
    x_check = x[:, :3, :] # 크기가 큰 3번째 coefficient까지만 고려

    mean = np.mean(x_check, axis=0, keepdims=True)
    std = np.std(x_check, axis=0, keepdims=True)
    z_score = np.abs((x_check - mean) / (std + 1e-8))
    outlier_mask = z_score > threshold
    
    bad_frames = np.any(outlier_mask, axis=(1, 2))
    valid_frames_mask = ~bad_frames
    frame_idx = np.arange(n_frames)
    valid_idx = frame_idx[valid_frames_mask]
    
    diff1 = np.diff(x, axis=0)
    kurtosis_diff1 = kurtosis(diff1, axis=0, fisher=True, nan_policy="omit")
    kurtosis_diff1 = np.nan_to_num(kurtosis_diff1, nan=0.0)
    
    if len(valid_idx) == n_frames:
        return x

    if len(valid_idx) > 4:
        interp_x = x.copy()
        for i in range(n_harmonics):
            for j in range(4):
                valid_vals = x[:, i, j]
                valid_vals = valid_vals[valid_frames_mask]
                if kurtosis_diff1[i, j] > pulse_threshold:
                    f = PchipInterpolator(valid_idx, valid_vals, axis=0, extrapolate=True)
                else:
                    f = CubicSpline(valid_idx, valid_vals, axis=0, bc_type='natural', extrapolate=True)
                missing_idx = frame_idx[~valid_frames_mask]
                inside = ((missing_idx >= valid_idx[0]) & (missing_idx <= valid_idx[-1]))
                interp_idx = missing_idx[inside]
                interp_x[interp_idx, i, j] = f(interp_idx)
    else:
        interp_x = x

    return interp_x

def get_periodicity_mask(label_cfgs, autocorr: np.ndarray) -> np.ndarray:
    _, harmonics, coeffs = autocorr.shape
    mask = np.zeros((harmonics, coeffs), dtype=bool)

    for i in range(harmonics):
        for j in range(coeffs):
            signal = autocorr[1:, i, j] #Lag 0 제거
            signal = np.abs(signal)
            peaks, properties = find_peaks(signal, height=label_cfgs['min_peak_height'], 
                                           distance=label_cfgs['min_peak_dist'], 
                                           prominence=label_cfgs['peak_prominence'])
            if len(peaks) >= 3:
                intervals = np.diff(peaks)
                median_interval = np.median(intervals)
                relative_error = np.abs(intervals - median_interval) / median_interval
                is_periodic = np.mean(relative_error <= 0.2) > 0.8
                is_high = np.mean(properties["peak_heights"]) >= 0.4
                mask[i, j] = is_periodic & is_high
    return mask

def check_drop_mask(label_cfgs, x: np.ndarray):
    # (f, h, 4)
    x = np.asarray(x, dtype=float)

    if x.ndim != 3:
        raise ValueError(f"Expected (frame, h, 4), got {x.shape}")

    x_shifted = x - np.min(x, axis=0, keepdims=True)
    baseline = np.percentile(x_shifted, 90, axis=0)
    is_drop = x_shifted < baseline * label_cfgs["zero_ratio"]

    min_len = label_cfgs["min_drop_len"]
    mask = np.zeros(x.shape[1:], dtype=bool)

    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            signal = is_drop[:, i, j]

            # True 구간의 시작과 끝 탐지
            edges = np.diff(np.concatenate(([False], signal, [False])).astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)

            lengths = ends - starts

            mask[i, j] = np.any(lengths >= min_len)

    return mask

def make_label(series: np.ndarray, corr_score: np.ndarray, corr_mask: np.ndarray, target_shape: tuple, part_name: list, cfgs) -> Diagnosis | None:
    label_cfgs = cfgs['label']
    f, p, h, c = target_shape
    if series.shape != target_shape: return None
    label = np.full((p, h, c), "", dtype=object)
    periodicity = np.zeros((f, p, h, c))
    new_series = series.copy()
    
    if len(corr_score) != p or len(corr_mask) != p:
        raise ValueError(f"Part count mismatch: expected {p}, got {len(corr_score)} len score & {len(corr_mask)} len mask")

    for i in range(p):
        part_corr = abs(corr_score[i]) # 0 ~ 1 사이
        part_mask = corr_mask[i] # boolean mask
        data = series[:, i, :, :] # (f, h, c)

        if not part_mask: #마스크로 제외된 경우 중 대부분 값이 0인 경우
            zero_ratio = np.mean(data==0)
            if zero_ratio >= 0.8:
                part_label = np.full((h, c), "zero", dtype=object)
                label[i] = part_label
                continue
        
        drop_mask = np.zeros((h, c), dtype=bool)
        if part_name[i] not in ("left_arm", "right_arm"): #규칙적인 drop이 없는 게 확실한 경우 outlier 제거 우선
            noise_threshold = label_cfgs['outlier_thresh'] + part_corr #threshold를 2.5-3 사이로 조정
            data = interpolate_outlier(data, threshold=noise_threshold, pulse_threshold=noise_threshold, n_frames=f, n_harmonics=h)
            new_series[:, i] = data
        else:
            drop_mask = check_drop_mask(label_cfgs=label_cfgs, x=data)
        autocorr = cal_autocorr(x=data, dir=cfgs['autocorr_axis'], n_frames=f) # (128, 30, 4)
        autocorr_mask = get_periodicity_mask(label_cfgs=label_cfgs, autocorr=autocorr) #(30, 4)

        nonzero_mask = label[i] != 'zero'
        label[i, nonzero_mask & ~autocorr_mask] = 'noise' # 주기성이 없다면 noise
        label[i, nonzero_mask & drop_mask] = 'drop' # 연속 drop이 있다면 drop
        label[i, nonzero_mask & ~drop_mask & autocorr_mask] = 'clean' # 주기성이 있고 연속 drop이 없다면 clean
        periodicity[:, i] = autocorr

    diagnosis = Diagnosis(periodicity=periodicity, label=label, data=new_series)
    return diagnosis

#------------------------------------------------------------------------
# Denoising 함수
# drop_interpolate: drop 데이터를 보간
# final_outlier: 일반 데이터를 보간
# denoising: 전체 denoising 
#------------------------------------------------------------------------  

def drop_interpolate(x: np.ndarray, window: int = 3, dense_thres: float = 0.2) -> np.ndarray:
    # x (f, N)
    x_interpolated = x.copy()
    frame, N = x.shape
    for i in range(N):
        col = x_interpolated[:, i]
        xmax = np.max(col)
        xmin = np.min(col)
        interval = (xmax - xmin) * 0.75
        diffs = np.abs(np.diff(col, prepend=col[0]))
        vibe_mask = diffs > interval # 급격한 값변화 masking

        true_indices = np.where(vibe_mask)[0] #값변화가 필요한 인덱스 체크
        if len(true_indices) < 2: continue

        gaps = np.abs(np.diff(true_indices, prepend=0)) #인덱스 간의 gap으로 구간 길이 확인 (중간이 없고 분산이 큰 형태)
        g_upper = np.max(np.diff(np.sort(gaps)))
        gmin = np.min(gaps)
        threshold = gmin + g_upper # 저점 끝과 고점 끝의 간격을 사용하여 고점이 포함되거나 저점이 포함되지 않는 경우 방지
        dense_pair = np.where(gaps < threshold)[0] # 저점 모음

        needed = true_indices[dense_pair] # 간격이 저점인 인덱스만 체크
        mask = np.zeros_like(vibe_mask, dtype=bool)
        mask[needed] = True # 간격이 짧은(진동) 구간 마스킹

        #보간
        padded = np.concatenate([[False], mask, [False]])
        kernel = np.ones(window) / window
        density = np.convolve(padded, kernel, mode='same')
        padded = density > dense_thres
        
        diff_mask = np.diff(padded.astype(np.int8))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0] - 1

        drop_idx = np.where(vibe_mask)[0]
        for start, end in zip(starts, ends):
            if start == 0 or end == frame - 1: continue

            dmin = np.min(drop_idx)
            dmax = np.max(drop_idx)

            if dmin < start: left = np.max(drop_idx[drop_idx < start]) + 1
            else: left = start - 1
            if dmax > end: right = np.min(drop_idx[drop_idx > end]) - 1
            else: right = end + 1

            l = start - left
            r = right - end
            if r < l: val = col[right]
            else: val = col[left]

            col[start:end + 1] = val

    return x_interpolated

def final_outlier(x: np.ndarray, threshold: float = 3.0, pulse_threshold: float = 2.0) -> np.ndarray:
    # x  (f, N)
    x_interpolated = x.copy().astype(float)
    frames, N = x.shape
    diff1 = np.diff(x, axis=0)
    kurtosis_diff1 = kurtosis(diff1, axis=0, fisher=True, nan_policy="omit")
    kurtosis_diff1 = np.nan_to_num(kurtosis_diff1, nan=0.0)

    for i in range(N):
        col_data = x_interpolated[:, i]

        mean = np.mean(col_data)
        std = np.std(col_data)
        z_score = np.abs((col_data - mean) / (std + 1e-8))
        outlier_mask = z_score > threshold

        valid_frames_mask = ~outlier_mask
        frame_idx = np.arange(frames)
        valid_idx = frame_idx[valid_frames_mask]
        
        if len(valid_idx) == frames: continue
        if len(valid_idx) > 4:
            valid_vals = col_data[valid_frames_mask]
            if kurtosis_diff1[i] > pulse_threshold:
                f = PchipInterpolator(valid_idx, valid_vals, extrapolate=True)
            else:
                f = CubicSpline(valid_idx, valid_vals, bc_type='natural', extrapolate=True)
            missing_idx = frame_idx[~valid_frames_mask]
            inside = ((missing_idx >= valid_idx[0]) & (missing_idx<= valid_idx[-1]))
            interp_idx = missing_idx[inside]
            col_data[interp_idx] = f(interp_idx)
        else: continue

    return x_interpolated

def denoising(cfgs, diagnosis: Diagnosis) -> np.ndarray:
    denoise_cfgs = cfgs['denoise']
    periodicity = diagnosis.periodicity
    label = diagnosis.label
    data = diagnosis.data

    dropmask = (label == 'drop')
    if np.any(dropmask):
        data[:, dropmask] = drop_interpolate(data[:, dropmask], 
                                             window=denoise_cfgs['interpol_window'], dense_thres=denoise_cfgs['dense_thres'])

    noisemask = (label == 'noise')
    if np.any(noisemask):
        data[:, noisemask] = final_outlier(data[:, noisemask], threshold=denoise_cfgs['noise_thres'])

    cleanmask = (label == 'clean')
    if np.any(cleanmask):
        data[:, cleanmask] = final_outlier(data[:, cleanmask], threshold=denoise_cfgs['clean_thres'])

    return data

'''
#------------------------------------------------------------------------
# main code
#------------------------------------------------------------------------
def run_pipeline(tensor: np.ndarray):
    series, corr_score, corr_mask = find_best_window(tensor) # (frame, 8, 30, 4) -> (128, 8, 30, 4), 자가상관 점수 (8, ), part mask (8, )
    diagnosis = make_label(series, corr_score, corr_mask) # trier, periodicity, label, data
    interp_data = denoising(diagnosis)
    diagnosis.data = interp_data
    return diagnosis

if __name__ == "__main__":
    for name in os.listdir(EFD_PATH):
        name_path = os.path.join(EFD_PATH, name)
        for root, dirs, files in os.walk(name_path):
            for file in files:
                if not file.endswith(".npy"): continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, name_path)
                dest_dir = os.path.join(DEN_PATH, name, rel_path)
                os.makedirs(dest_dir, exist_ok = True)
                filename = os.path.splitext(file)[0]
                dest_path = os.path.join(dest_dir, f"{filename}.npy")

                efd = np.load(file_path)
                result = run_pipeline(efd)

                np.save(dest_path, result.data)
                print(f"========== [SAVED] {dest_path} ==========")
                print(f" final shape : {result.data.shape}")
                print("# part name :  score")
                for key, value in PART_NAMES.items():
                    corr = result.periodicity[1:, key, :, :]
                    score = np.mean(np.max(np.abs(corr), axis=0))
                    print(f"# {value:<10}: {score:>6.4f}")
                print("\n")
'''