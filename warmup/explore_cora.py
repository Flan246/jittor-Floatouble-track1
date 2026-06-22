"""阶段 A：仅探查 cora.pkl，不依赖 Jittor。"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

PKL = Path(__file__).resolve().parent / "data" / "cora.pkl"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "数据核查报告.md"

SPEC = {
    "num_nodes": 2708,
    "num_edges": 5429,
    "num_features": 1433,
    "num_classes": 7,
}

EXPECTED_KEYS = {
    "x",
    "y",
    "edge_index",
    "train_mask",
    "val_mask",
    "test_mask",
}


def _as_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "__dict__"):
        return vars(raw)
    raise TypeError(f"不支持的容器类型: {type(raw)}")


def _mask_sum(mask: np.ndarray) -> int:
    if mask.dtype == bool:
        return int(mask.sum())
    return int((mask != 0).sum())


def _check(name: str, ok: bool, detail: str) -> str:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return f"| {name} | {SPEC.get(name, '—')} | {detail} | {status} |"


def report_container_type(raw: dict) -> list[str]:
    lines = []
    print(f"\n=== 容器 ===\ntype(raw) = {type(raw).__name__}")
    print(f"keys = {sorted(raw.keys())}")
    lines.append(f"- 类型: `{type(raw).__name__}`")
    lines.append(f"- keys: `{sorted(raw.keys())}`")
    missing = EXPECTED_KEYS - set(raw.keys())
    extra = set(raw.keys()) - EXPECTED_KEYS - {"num_classes", "num_features"}
    if missing:
        lines.append(f"- ⚠ 缺少字段: `{sorted(missing)}`")
    if extra:
        lines.append(f"- 额外字段: `{sorted(extra)}`")
    return lines


def report_fields(raw: dict) -> tuple[list[str], dict]:
    lines = []
    stats: dict[str, Any] = {}
    print("\n=== 字段 shape / dtype ===")
    lines.append("| 字段 | dtype | shape | 备注 |")
    lines.append("|------|-------|-------|------|")

    for key in sorted(raw.keys()):
        val = raw[key]
        if isinstance(val, (int, float, np.integer)):
            note = str(val)
            lines.append(f"| `{key}` | scalar | — | {note} |")
            stats[key] = val
            print(f"  {key}: scalar = {val}")
            continue
        if not isinstance(val, np.ndarray):
            note = type(val).__name__
            lines.append(f"| `{key}` | {note} | — | 非 ndarray |")
            continue
        arr = val
        note = ""
        if key == "x":
            note = f"min={arr.min():.4f}, max={arr.max():.4f}, nnz_ratio={(arr != 0).mean():.4f}"
            stats["num_nodes"] = arr.shape[0]
            stats["num_features"] = arr.shape[1] if arr.ndim == 2 else None
        if key == "edge_index":
            note = f"row0∈[0,{arr[1].max() if arr.size else 0}], row1 max={arr[1].max() if arr.size else 0}"
            stats["num_edges_index"] = arr.shape[1] if arr.ndim == 2 else None
        if key.endswith("_mask"):
            note = f"sum={_mask_sum(arr)}"
        if key == "y":
            note = f"unique={np.unique(arr)}"
        lines.append(f"| `{key}` | `{arr.dtype}` | `{arr.shape}` | {note} |")
        print(f"  {key}: dtype={arr.dtype}, shape={arr.shape} {note}")
        stats[key] = arr

    return lines, stats


def report_masks(raw: dict) -> list[str]:
    lines = []
    tm = np.asarray(raw["train_mask"]).astype(bool)
    vm = np.asarray(raw["val_mask"]).astype(bool)
    xm = np.asarray(raw["test_mask"]).astype(bool)
    n = int(np.asarray(raw["x"]).shape[0])

    inter_tv = (tm & vm).sum()
    inter_tt = (tm & xm).sum()
    inter_vt = (vm & xm).sum()
    union = (tm | vm | xm).sum()

    print("\n=== mask 划分 ===")
    print(f"  train={tm.sum()}, val={vm.sum()}, test={xm.sum()}, N={n}")
    print(f"  交集 train∩val={inter_tv}, train∩test={inter_tt}, val∩test={inter_vt}")
    print(f"  并集={union}, 互斥且覆盖全部: {union == n and inter_tv == inter_tt == inter_vt == 0}")

    lines.append(f"- train: **{tm.sum()}** ({100 * tm.sum() / n:.1f}%)")
    lines.append(f"- val: **{vm.sum()}** ({100 * vm.sum() / n:.1f}%)")
    lines.append(f"- test: **{xm.sum()}** ({100 * xm.sum() / n:.1f}%)")
    lines.append(f"- 三 mask 互斥: **{'是' if inter_tv + inter_tt + inter_vt == 0 else '否'}**")
    if union == n:
        union_note = "是"
    else:
        union_note = f"否 (并集={union}, N={n})；官方包常见，仅对有 mask 的节点监督"
    lines.append(f"- 并集=全部节点: **{union_note}**")
    return lines


def report_labels(raw: dict) -> tuple[list[str], bool]:
    lines = []
    y = np.asarray(raw["y"])
    tm = np.asarray(raw["train_mask"]).astype(bool)
    vm = np.asarray(raw["val_mask"]).astype(bool)
    xm = np.asarray(raw["test_mask"]).astype(bool)

    y_test = y[xm]
    y_train = y[tm]
    test_all_minus1 = bool(np.all(y_test == -1))

    print("\n=== 标签 ===")
    print(f"  y[test_mask] unique: {np.unique(y_test)}")
    print(f"  测试集全为 -1: {test_all_minus1}")
    print(f"  y[train_mask] unique: {np.unique(y_train)}")

    lines.append(f"- 测试集 `y` 是否全为 **-1**: **{'是' if test_all_minus1 else '否'}**")
    lines.append(f"- 训练集标签取值: `{np.unique(y_train).tolist()}`")
    if len(y_train):
        hist = {int(c): int((y_train == c).sum()) for c in np.unique(y_train)}
        lines.append(f"- 训练集类别直方图: `{hist}`")
    return lines, test_all_minus1


def compare_with_spec(raw: dict, stats: dict) -> tuple[list[str], bool]:
    lines = ["| 核查项 | 赛题期望 | 实测 | 结果 |", "|--------|----------|------|------|"]
    all_ok = True

    x = np.asarray(raw["x"])
    ei = np.asarray(raw["edge_index"])
    y = np.asarray(raw["y"])
    nc = raw.get("num_classes", len(np.unique(y[y >= 0])))
    nf = raw.get("num_features", x.shape[1] if x.ndim == 2 else None)

    checks = [
        ("num_nodes", x.shape[0] == SPEC["num_nodes"], f"x.shape[0]={x.shape[0]}"),
        ("num_features", nf == SPEC["num_features"], f"num_features={nf}"),
        (
            "num_edges (无向引用)",
            True,
            f"edge_index.shape[1]={ei.shape[1]}（双向边列数常为 2×5429）",
        ),
        ("num_classes", int(nc) == SPEC["num_classes"], f"num_classes={nc}"),
    ]

    for name, ok, detail in checks:
        if name == "num_edges (无向引用)":
            # 5429 为无向边数；edge_index 常为双向 10556 或单向 5429
            e = ei.shape[1]
            ok = e in (SPEC["num_edges"], 2 * SPEC["num_edges"])
            detail = f"edge_index.shape[1]={e}"
        lines.append(_check(name, ok, detail))
        all_ok = all_ok and ok

    return lines, all_ok


def save_summary_md(sections: list[str]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(sections)
    REPORT.write_text(body, encoding="utf-8")
    print(f"\n已写入: {REPORT}")


def main() -> None:
    if not PKL.is_file():
        raise FileNotFoundError(
            f"未找到 {PKL}。请先运行: python warmup/scripts/build_cora_pkl.py "
            "或从头歌下载评测包放到 warmup/data/cora.pkl"
        )

    with open(PKL, "rb") as f:
        raw = _as_dict(pickle.load(f))

    sec1 = ["# Cora 数据核查报告", "", f"数据文件: `{PKL}`", ""]
    sec1 += ["## 1. 文件与容器说明", ""] + report_container_type(raw)

    field_lines, stats = report_fields(raw)
    sec2 = ["## 2. 各字段 dtype 与 shape", ""] + field_lines

    sec3 = ["## 3. train / val / test 划分", ""] + report_masks(raw)

    label_lines, test_ok = report_labels(raw)
    sec4 = ["## 4. 标签分布", ""] + label_lines

    spec_lines, spec_ok = compare_with_spec(raw, stats)
    sec5 = ["## 5. 与赛题规格对照", ""] + spec_lines

    # 官方 cora：mask 并集可小于 N（1068 节点无划分），不阻断阶段 B
    can_proceed = spec_ok and test_ok
    sec6 = [
        "## 6. 结论",
        "",
        f"- **可否进入阶段 B（gcn.py 加载与训练）**: **{'是' if can_proceed else '否（请先核对 FAIL 项）'}**",
        "- 特征已做 `NormalizeFeatures`（Planetoid 导出时）；训练时无需重复归一化。",
        "- **人工签字（A3 闸门）**: 请对照 [竞赛备忘.md](竞赛备忘.md) 与头歌赛题页，确认后在本节注明日期。",
        "",
        "| 确认人 | 日期 | 备注 |",
        "|--------|------|------|",
        "| （待填写） | | 数据与赛题一致 |",
    ]

    save_summary_md([*sec1, *sec2, *sec3, *sec4, *sec5, *sec6])


if __name__ == "__main__":
    main()
