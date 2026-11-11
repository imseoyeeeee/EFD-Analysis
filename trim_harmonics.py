"""
Keep only the first N harmonics of already extracted EFD pickles (in place).

EFD harmonics are computed independently, so the first N of a 128-harmonic file are identical
to a file extracted with harmonics_num: N. Files that cannot be read (e.g. the one being written
when the run was stopped) are deleted so that `skip_existing: true` extracts them again.

python EFD-Analysis/trim_harmonics.py --root <output_path>/EFD8-unnorm --harmonics 32
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path, help='EFD folder, e.g. <output_path>/EFD8-unnorm')
    ap.add_argument('--harmonics', required=True, type=int)
    args = ap.parse_args()

    n_trim = n_ok = n_bad = 0
    for tmp in args.root.rglob('*.pkl.tmp'):
        tmp.unlink()
    for i, path in enumerate(sorted(args.root.rglob('*.pkl')), 1):
        try:
            with open(path, 'rb') as f:
                data = np.asarray(pickle.load(f))
            assert data.ndim == 4 and data.shape[0] > 0
        except Exception as e:
            print('[deleted] unreadable:', path, type(e).__name__)
            path.unlink()
            n_bad += 1
            continue
        if data.shape[2] < args.harmonics:
            raise ValueError('{} has only {} harmonics'.format(path, data.shape[2]))
        if data.shape[2] == args.harmonics:
            n_ok += 1
            continue
        tmp = path.with_suffix('.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(np.ascontiguousarray(data[:, :, :args.harmonics]), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        n_trim += 1
        if i % 500 == 0:
            print('[{} files checked]'.format(i))
    print('done: {} trimmed to {} harmonics, {} already {}, {} unreadable deleted'.format(
        n_trim, args.harmonics, n_ok, args.harmonics, n_bad))


if __name__ == '__main__':
    main()
