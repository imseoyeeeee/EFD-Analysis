"""
Sanity check of extracted EFD pickles: were they really made from human-parsing label maps?

Picks a few random .pkl files and reports, per file and per part, how often the part is present
(non-zero EFD), and flags typical mistakes:
  * binary silhouette 0/255 as input -> only the full-body part exists, every other part is empty
  * binary silhouette 0/1   as input -> the "head" contour is the whole body (same as full body)
  * colour / palette images         -> parts look random (very unstable presence)
  * unreadable / empty / NaN files, wrong shape

python EFD-Analysis/data_check.py --cfgs EFD-Analysis/configs/fineefd.yaml          # EFD folder from the config
python EFD-Analysis/data_check.py --root <output_path>/EFD8-unnorm --num 50         # or give the folder
"""
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import yaml

FINE_PARTS = ['background', 'head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg', 'dress']


def efd_dirname(efd_cfgs):
    n = len(efd_cfgs["parts"])
    return f"EFD{n}" if efd_cfgs.get("normalize", False) else f"EFD{n}-unnorm"


def load(path):
    with open(path, 'rb') as f:
        return np.asarray(pickle.load(f))


def check_file(data, n_parts):
    """-> dict of per-file statistics, or an error string."""
    if data.ndim != 4 or data.shape[-1] != 4:
        return 'unexpected shape {} (expected [T, P, H, 4])'.format(data.shape)
    if data.shape[0] == 0:
        return 'empty (0 frames)'
    if data.shape[1] != n_parts:
        return '{} parts, config has {}'.format(data.shape[1], n_parts)
    if not np.isfinite(data).all():
        return 'contains NaN / inf'
    present = np.abs(data).sum(axis=(2, 3)) > 0                       # [T, P]
    # head (part 1) identical to the full body (part 0) -> silhouette 0/1 read as label 1
    both = present[:, 0] & present[:, 1] if n_parts > 1 else np.zeros(len(data), bool)
    if both.any():
        diff = np.linalg.norm(data[both, 1] - data[both, 0], axis=(1, 2))
        ref = np.linalg.norm(data[both, 0], axis=(1, 2)) + 1e-8
        head_is_body = float(np.mean(diff / ref < 0.05))
    else:
        head_is_body = 0.
    # presence flips between consecutive frames (random labels -> very unstable)
    flips = np.mean(present[1:] != present[:-1]) if len(data) > 1 else 0.
    return {'frames': data.shape[0], 'harmonics': data.shape[2], 'presence': present.mean(0),
            'head_is_body': head_is_body, 'flips': float(flips)}


def main():
    ap = argparse.ArgumentParser(description='Check extracted EFD pickles')
    ap.add_argument('--cfgs', type=str, default='', help='EFD-Analysis config (EFD folder + part names)')
    ap.add_argument('--root', type=str, default='', help='EFD folder, e.g. <output_path>/EFD8-unnorm')
    ap.add_argument('--num', type=int, default=20, help='number of random files to check')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    names = None
    root = Path(args.root) if args.root else None
    if args.cfgs:
        cfg = yaml.safe_load(open(args.cfgs))
        names = list(cfg['efd_config']['parts'].keys())
        if root is None:
            root = Path(cfg['paths']['output_path']) / efd_dirname(cfg['efd_config'])
    if root is None:
        ap.error('give --cfgs or --root')
    if not root.is_dir():
        raise FileNotFoundError('EFD folder not found: {}'.format(root))

    files = sorted(root.rglob('*.pkl'))
    if not files:
        raise FileNotFoundError('no .pkl file in {}'.format(root))
    random.seed(args.seed)
    picked = sorted(random.sample(files, min(args.num, len(files))))
    if names is None:                                    # part count from the first readable file
        n = None
        for path in picked:
            try:
                data = load(path)
            except Exception:
                continue
            if data.ndim == 4:
                n = data.shape[1]
                break
        n = len(FINE_PARTS) if n is None else n
        names = FINE_PARTS if n == len(FINE_PARTS) else ['part{}'.format(i) for i in range(n)]
    P = len(names)

    print('EFD folder : {}'.format(root))
    print('files      : {} found, {} checked\n'.format(len(files), len(picked)))
    short = [nm[:6] for nm in names]
    print('{:<44} {:>6} {:>4}  '.format('file', 'frames', 'H') + ' '.join('{:>6}'.format(s) for s in short))
    print('{:<44} {:>6} {:>4}  '.format('', '', '') + '(part presence: fraction of frames with a contour)')

    stats, errors = [], []
    for path in picked:
        rel = str(path.relative_to(root))
        rel = rel if len(rel) <= 44 else '...' + rel[-41:]
        try:
            r = check_file(load(path), P)
        except Exception as e:                                   # unreadable / truncated pickle
            r = 'unreadable ({})'.format(type(e).__name__)
        if isinstance(r, str):
            errors.append((path, r))
            print('{:<44} ERROR: {}'.format(rel, r))
            continue
        stats.append(r)
        print('{:<44} {:>6} {:>4}  '.format(rel, r['frames'], r['harmonics'])
              + ' '.join('{:>6.2f}'.format(p) for p in r['presence']))

    print()
    if not stats:
        print('RESULT: no readable file - check the folder and the files above.')
        return
    presence = np.mean([r['presence'] for r in stats], axis=0)
    head_is_body = np.mean([r['head_is_body'] for r in stats])
    flips = np.mean([r['flips'] for r in stats])
    harmonics = sorted({r['harmonics'] for r in stats})
    print('mean presence  : ' + ', '.join('{} {:.2f}'.format(nm, p) for nm, p in zip(names, presence)))
    print('harmonics      : {}'.format(harmonics))
    print('head == body   : {:.2f} of frames (head contour identical to the full body)'.format(head_is_body))
    print('presence flips : {:.2f} (fraction of part on/off changes between consecutive frames)'.format(flips))
    print()

    problems = []
    if errors:
        problems.append('{} file(s) unreadable or invalid (see ERROR lines). '
                        'Delete them and re-extract with skip_existing: true.'.format(len(errors)))
    if len(harmonics) > 1:
        problems.append('files have different harmonic counts {}: do not mix them in one folder '
                        '(trim with trim_harmonics.py).'.format(harmonics))
    others = presence[1:] if P > 1 else np.array([1.])
    if presence[0] > 0.5 and others.max() < 0.05:
        problems.append('only the full-body part has contours -> the input was most likely a BINARY '
                        'SILHOUETTE (0/255), not a parsing label map.')
    elif head_is_body > 0.5 and np.delete(others, 0).max() < 0.05:
        problems.append('the head contour equals the full body and the other parts are empty -> the input '
                        'was most likely a BINARY SILHOUETTE with values 0/1 (read as label 1 = head).')
    elif flips > 0.3:
        problems.append('parts switch on/off very often between frames -> the label values may not be '
                        'parsing ids (colour / palette images?). Check np.unique of a few input PNGs.')
    if presence[0] < 0.5:
        problems.append('the full body is missing in {:.0%} of the frames -> many empty / wrong input '
                        'images?'.format(1 - presence[0]))

    if problems:
        print('RESULT: POSSIBLE PROBLEMS')
        for p in problems:
            print('  - ' + p)
    else:
        print('RESULT: OK - looks like EFD extracted from parsing label maps '
              '(some parts can be missing in some frames, e.g. dress or occluded arms).')


if __name__ == '__main__':
    main()
