"""
原始日志 → 清洗 → 时间排序 → ID 编码 → 时序边表 → （可选）切分

切分策略由 configs/default.yaml 的 split.strategy 控制；为 null 时只写全量 processed，不生成 val。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "track1" / "configs" / "default.yaml"

# 与 configs/default.yaml 同步；无 PyYAML 时使用
DEFAULT_CFG: dict = {
    "paths": {
        "raw_data": "track1/data",
        "processed": "track1/processed",
        "outputs": "track1/outputs",
        "checkpoints": "track1/checkpoints",
    },
    "datasets": ["dataset1", "dataset2"],
    "preprocess": {
        "drop_duplicate_edges": True,
        "drop_self_loops": False,
        "sort_by_time": True,
        "remap_ids": True,
    },
    "split": {"strategy": None, "val_ratio": 0.15},
}


def load_config(path: Path) -> dict:
    if yaml is not None and path.is_file():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    if path.is_file():
        raise ImportError(
            "未安装 PyYAML，无法读取 YAML 配置。请执行:\n"
            "  D:\\miniconda3\\envs\\jittorgeometric\\python.exe -m pip install PyYAML\n"
            "或省略 --config，使用内置 DEFAULT_CFG。"
        )
    return DEFAULT_CFG.copy()


def clean_edges(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    if cfg["preprocess"]["drop_duplicate_edges"]:
        keys = [c for c in ["src", "dst", "time"] if c in out.columns]
        out = out.drop_duplicates(subset=keys, keep="first")
    if cfg["preprocess"]["drop_self_loops"] and "src" in out.columns and "dst" in out.columns:
        out = out[out.src != out.dst]
    return out


def sort_edges(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not cfg["preprocess"]["sort_by_time"]:
        return df
    cols = ["time"]
    if "edge_id" in df.columns:
        cols.append("edge_id")
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def remap_ids(train: pd.DataFrame, test: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """连续编码；test 候选列沿用训练期映射，未知 ID 记为 -1（后续模型需处理）。"""
    maps: dict = {"dataset": dataset, "src": {}, "dst": {}}
    tr = train.copy()
    te = test.copy()

    def fit_map(col: str, values: pd.Series) -> dict:
        uniq = pd.Index(values.unique())
        return {int(v): i for i, v in enumerate(uniq)}

    tr["src"] = tr["src"].astype("int64")
    tr["dst"] = tr["dst"].astype("int64")
    maps["src"] = fit_map("src", tr["src"])
    maps["dst"] = fit_map("dst", tr["dst"])

    tr["src"] = tr["src"].map(maps["src"])
    tr["dst"] = tr["dst"].map(maps["dst"])

    te["src"] = te["src"].map(maps["src"]).fillna(-1).astype("int64")
    cand_cols = [c for c in te.columns if c.startswith("c") and c[1:].isdigit()]
    for c in cand_cols:
        te[c] = te[c].map(maps["dst"]).fillna(-1).astype("int64")

    maps["num_src"] = len(maps["src"])
    maps["num_dst"] = len(maps["dst"])
    return tr, te, maps


def apply_split(train: pd.DataFrame, dataset: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    split_cfg = cfg.get("split") or {}
    strategy = split_cfg.get("strategy")
    if strategy is None:
        return train, None

    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    if strategy == "row_ratio":
        n = len(train)
        cut = int(n * (1 - val_ratio))
        return train.iloc[:cut].copy(), train.iloc[cut:].copy()
    if strategy == "time_ratio":
        tr = train.sort_values("time", kind="mergesort")
        cut = int(len(tr) * (1 - val_ratio))
        return tr.iloc[:cut].copy(), tr.iloc[cut:].copy()
    if strategy == "official_split":
        if "split" not in train.columns:
            raise ValueError(f"{dataset} 无 split 列，不能使用 official_split")
        tr_train = train[train["split"] == 0].drop(columns=["split"])
        tr_val = train[train["split"] == 1].drop(columns=["split"])
        return tr_train.copy(), tr_val.copy()

    raise ValueError(f"未知 split.strategy: {strategy!r}")


def run_dataset(dataset: str, cfg: dict, config_path: Path) -> None:
    raw_root = (REPO_ROOT / cfg["paths"]["raw_data"]).resolve()
    out_root = (REPO_ROOT / cfg["paths"]["processed"]).resolve() / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(raw_root / dataset / "train.csv")
    test = pd.read_csv(raw_root / dataset / "test.csv")
    manifest: dict = {"dataset": dataset, "config": str(config_path), "steps": {}}

    n0 = len(train)
    train = clean_edges(train, cfg)
    manifest["steps"]["clean"] = {"train_rows_before": n0, "train_rows_after": len(train)}

    train = sort_edges(train, cfg)
    train["edge_id"] = range(1, len(train) + 1)
    manifest["steps"]["sort"] = {"sorted_by_time": cfg["preprocess"]["sort_by_time"]}

    id_maps = None
    if cfg["preprocess"]["remap_ids"]:
        train, test, id_maps = remap_ids(train, test, dataset)
        manifest["steps"]["remap_ids"] = id_maps

    split_cfg = cfg.get("split") or {}
    tr_part, val_part = apply_split(train, dataset, cfg)
    manifest["steps"]["split"] = {
        "strategy": split_cfg.get("strategy"),
        "train_rows": len(tr_part),
        "val_rows": len(val_part) if val_part is not None else 0,
    }

    tr_part.to_csv(out_root / "train_edges.csv", index=False)
    if val_part is not None:
        val_part.to_csv(out_root / "val_edges.csv", index=False)
    test.to_csv(out_root / "test_queries.csv", index=False)

    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[{dataset}] processed → {out_root}")
    print(f"  train_edges: {len(tr_part):,}" + (f"  val_edges: {len(val_part):,}" if val_part is not None else "  (no val split)"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dataset1", "dataset2"])
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 配置路径；默认使用 configs/default.yaml（需 PyYAML）或内置 DEFAULT_CFG",
    )
    args = parser.parse_args()
    config_path = args.config.resolve() if args.config else DEFAULT_CONFIG
    if yaml is None and (args.config is None or not config_path.is_file()):
        cfg = DEFAULT_CFG.copy()
        config_path = DEFAULT_CONFIG
        print("提示: 未安装 PyYAML，使用内置 DEFAULT_CFG。安装后可读 YAML: pip install PyYAML")
    else:
        cfg = load_config(config_path)
    run_dataset(args.dataset, cfg, args.config)


if __name__ == "__main__":
    main()
