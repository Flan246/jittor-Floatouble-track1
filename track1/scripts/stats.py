"""A 榜数据快速概览（读 track1/data → data_A）。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "track1" / "data"


def summarize(dataset: str, data_dir: Path) -> None:
    train_path = data_dir / dataset / "train.csv"
    test_path = data_dir / dataset / "test.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"缺少 {train_path}，请确认 track1/data 已联接至 data_A")

    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    print(f"\n=== {dataset} ===")
    print(f"train: {len(tr):,}  rows  columns={list(tr.columns)}")
    print(f"test:  {len(te):,}  rows")
    print(f"src [{tr.src.min()}, {tr.src.max()}]  dst [{tr.dst.min()}, {tr.dst.max()}]")
    print(f"time [{tr.time.min()}, {tr.time.max()}]")
    tdiff = tr.time.diff().dropna()
    print(f"time non-decreasing: {(tdiff >= 0).all()}")
    dup = tr.duplicated(subset=[c for c in ["src", "dst", "time"] if c in tr.columns]).sum()
    print(f"duplicate (src,dst,time): {dup:,}")
    if "split" in tr.columns:
        print("split counts:")
        print(tr["split"].value_counts().sort_index().to_string())
    src_u, dst_u = set(tr.src.unique()), set(tr.dst.unique())
    print(f"unique src={len(src_u):,}  dst={len(dst_u):,}  overlap={len(src_u & dst_u):,}")
    self_loops = (tr.src == tr.dst).sum()
    print(f"self-loops (src==dst): {self_loops:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "dataset1", "dataset2"])
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"数据目录不存在: {data_dir}")
        print("请执行: cmd /c mklink /J track1\\data data_A")
        raise SystemExit(1)

    names = [args.dataset] if args.dataset else ["dataset1", "dataset2"]
    for name in names:
        summarize(name, data_dir)


if __name__ == "__main__":
    main()
