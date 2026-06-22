"""从 JittorGeometric 已处理 Cora 导出赛题格式 cora.pkl（无需 pandas）。"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "JittorGeometric" / "data" / "cora" / "processed" / "data.pkl"
OUT = Path(__file__).resolve().parents[1] / "data" / "cora.pkl"


def _normalize_features(x: np.ndarray) -> np.ndarray:
    row_sum = np.maximum(x.sum(axis=1, keepdims=True), 1.0)
    return (x / row_sum).astype(np.float32)


def main() -> None:
    os.environ.setdefault("JITTOR_HOME", r"D:\jittor_cache")
    import jittor as jt

    jt.flags.use_cuda = 0
    data, _slices = jt.load(str(PROCESSED))

    x = _normalize_features(np.asarray(data.x.numpy(), dtype=np.float32))
    y = np.asarray(data.y.numpy(), dtype=np.int64)
    edge_index = np.asarray(data.edge_index.numpy(), dtype=np.int64)
    train_mask = np.asarray(data.train_mask.numpy(), dtype=bool)
    val_mask = np.asarray(data.val_mask.numpy(), dtype=bool)
    test_mask = np.asarray(data.test_mask.numpy(), dtype=bool)

    y_out = y.copy()
    y_out[test_mask] = -1

    raw = {
        "x": x,
        "y": y_out,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_classes": 7,
        "num_features": int(x.shape[1]),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {OUT}")
    print(
        f"N={x.shape[0]}, F={x.shape[1]}, E={edge_index.shape[1]}, "
        f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
    )


if __name__ == "__main__":
    main()
