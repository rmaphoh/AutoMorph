"""元の AutoMorph の M2 の出力から、AutoMorph_fork の M3 の指標を計算して CSV に書く。

  python AutoMorph_fork/run_m3_fork.py --results Results --out automorph_fork_m3.csv

--results は読み込むだけで書き込まない。--limit N で先頭の N 眼だけを計算する。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parent / "M3_feature"))
from retipy import tortuosity_measures as tm  # noqa: E402

KINDS = {"vessel": ("binary_vessel", "binary"), "artery": ("artery_vein", "artery"), "vein": ("artery_vein", "vein")}
FULL = {"artery": "artery_binary_process", "vein": "vein_binary_process"}


def read_gray(path: Path) -> np.ndarray:
    img = io.imread(path)
    return img[..., 0] if img.ndim == 3 else img


def one_image(results: Path, name: str) -> dict:
    row = {"Name": name}
    m2 = results / "M2"
    for kind, (folder, label) in KINDS.items():
        skel = read_gray(m2 / folder / f"macular_centred_{label}_skeleton" / name)
        mask = read_gray(m2 / folder / f"macular_centred_{label}_process" / name)
        for key, value in tm.evaluate_whole(mask, skel).items():
            row[f"{kind}_{key}"] = value
    for zone in ("B", "C"):
        for kind in ("artery", "vein"):
            sk_path = m2 / "artery_vein" / f"macular_Zone_{zone}_centred_{kind}_skeleton" / name
            if sk_path.exists():
                row[f"zone{zone}_{kind}_cre"] = tm.evaluate_zone(read_gray(m2 / "artery_vein" / FULL[kind] / name), read_gray(sk_path), kind)["cre"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    names = sorted(p.name for p in (args.results / "M2" / "binary_vessel" / "macular_centred_binary_skeleton").glob("*.png"))[: args.limit]
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(one_image)(args.results, n) for n in names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out} ({len(rows)} images)")


if __name__ == "__main__":
    main()
