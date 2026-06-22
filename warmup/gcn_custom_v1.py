"""Cora GCN 节点分类 — 头歌热身赛提交脚本（评测机会重新运行本文件）。"""
from __future__ import annotations

import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import nn

from jittor_geometric.data import CSC, CSR
from jittor_geometric.nn import GCNConv
from jittor_geometric.nn.conv.gcn_conv import gcn_norm

ROOT = Path(__file__).resolve().parent
PKL_PATH = ROOT / "data" / "cora.pkl"
RESULT_JSON = ROOT / "result.json"
RESULT_ZIP = ROOT / "result.zip"

EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 5e-4
HIDDEN = 256
DROPOUT = 0.5


def setup_env() -> None:
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    jt.misc.set_global_seed(42)


def load_cora(pkl_path: str | Path = PKL_PATH) -> dict:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"期望 dict，得到 {type(raw)}")
    return raw


def _normalize_features(x: np.ndarray) -> np.ndarray:
    row_sum = x.sum(axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1.0)
    return (x / row_sum).astype(np.float32)


def to_jittor(raw: dict) -> dict:
    x_np = np.asarray(raw["x"], dtype=np.float32)
    if x_np.max() > 1.0 + 1e-3:
        x_np = _normalize_features(x_np)

    x = jt.array(x_np, dtype=jt.float32)
    y = jt.array(raw["y"], dtype=jt.int32)
    edge_index = jt.array(raw["edge_index"], dtype=jt.int32)
    train_mask = jt.array(raw["train_mask"].astype(bool))
    val_mask = jt.array(raw["val_mask"].astype(bool))
    test_mask = jt.array(raw["test_mask"].astype(bool))
    num_classes = int(raw.get("num_classes", 7))
    num_features = int(raw.get("num_features", x.shape[1]))
    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_classes": num_classes,
        "num_features": num_features,
    }


def _coo_to_csc_csr_numpy(
    edge_index: jt.Var, edge_weight: jt.Var, num_nodes: int
) -> tuple[CSC, CSR]:
    ei = edge_index.numpy()
    ew = np.asarray(edge_weight.numpy(), dtype=np.float32)
    src, dst = ei[0].astype(np.int64), ei[1].astype(np.int64)

    csc_order = np.argsort(dst, kind="stable")
    row_indices = src[csc_order].astype(np.int32)
    csc_w = ew[csc_order]
    col_off = np.zeros(num_nodes + 1, dtype=np.int32)
    np.add.at(col_off, dst + 1, 1)
    col_off = np.cumsum(col_off)

    csr_order = np.argsort(src, kind="stable")
    col_indices = dst[csr_order].astype(np.int32)
    csr_w = ew[csr_order]
    row_off = np.zeros(num_nodes + 1, dtype=np.int32)
    np.add.at(row_off, src + 1, 1)
    row_off = np.cumsum(row_off)

    csc = CSC(jt.array(row_indices), jt.array(col_off), jt.array(csc_w))
    csr = CSR(jt.array(col_indices), jt.array(row_off), jt.array(csr_w))
    return csc, csr


def build_adj(edge_index: jt.Var, edge_weight: jt.Var | None, num_nodes: int) -> tuple:
    edge_index, edge_weight = gcn_norm(
        edge_index,
        edge_weight,
        num_nodes,
        improved=False,
        add_self_loops=True,
    )
    with jt.no_grad():
        try:
            from jittor_geometric.ops import cootocsc, cootocsr

            csc = cootocsc(edge_index, edge_weight, num_nodes)
            csr = cootocsr(edge_index, edge_weight, num_nodes)
        except Exception:
            csc, csr = _coo_to_csc_csr_numpy(edge_index, edge_weight, num_nodes)
    return csc, csr


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        dropout: float,
        use_spmm: bool,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, spmm=use_spmm)
        self.conv2 = GCNConv(hidden, out_channels, spmm=use_spmm)
        self.dropout = dropout

    def execute(self, x, csc, csr):
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.is_training())
        return self.conv2(x, csc, csr)


def train_model(data: dict, csc, csr, use_spmm: bool) -> GCN:
    model = GCN(data["num_features"], HIDDEN, data["num_classes"], DROPOUT, use_spmm)
    optimizer = nn.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits = model(data["x"], csc, csr)
        loss = nn.cross_entropy_loss(logits[data["train_mask"]], data["y"][data["train_mask"]])
        optimizer.step(loss)

        model.eval()
        logits = model(data["x"], csc, csr)
        val_acc = _accuracy(logits, data["y"], data["val_mask"])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            train_acc = _accuracy(logits, data["y"], data["train_mask"])
            print(f"Epoch {epoch:03d} train_acc={train_acc:.4f} val_acc={val_acc:.4f} best_val={best_val:.4f}")

    if best_state is not None:
        model.load_parameters(best_state)
    print(f"Best val acc: {best_val:.4f}")
    return model


def _accuracy(logits: jt.Var, y: jt.Var, mask: jt.Var) -> float:
    pred, _ = jt.argmax(logits[mask], dim=1)
    return float(pred.equal(y[mask]).sum().item() / mask.sum().item())


def infer(model: GCN, data: dict, csc, csr) -> dict[str, int]:
    model.eval()
    logits = model(data["x"], csc, csr)
    pred_all, _ = jt.argmax(logits, dim=1)
    test_mask = data["test_mask"].numpy().astype(bool)
    test_ids = np.where(test_mask)[0]
    pred_np = pred_all.numpy()
    return {str(int(nid)): int(pred_np[nid]) for nid in test_ids}


def save_result_json(predictions: dict[str, int], path: Path = RESULT_JSON) -> None:
    ordered = {k: predictions[k] for k in sorted(predictions, key=int)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path} ({len(ordered)} nodes)")


def pack_submission(zip_path: Path = RESULT_ZIP) -> None:
    gcn_src = Path(__file__).resolve()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(gcn_src, arcname="gcn.py")
        zf.write(RESULT_JSON, arcname="result.json")
    print(f"Wrote {zip_path}")


def main() -> None:
    setup_env()
    use_spmm = bool(jt.flags.use_cuda)
    print(f"use_cuda={jt.flags.use_cuda}, spmm={use_spmm}")

    raw = load_cora()
    data = to_jittor(raw)
    num_nodes = data["x"].shape[0]
    csc, csr = build_adj(data["edge_index"], None, num_nodes)

    model = train_model(data, csc, csr, use_spmm)
    preds = infer(model, data, csc, csr)
    save_result_json(preds)
    pack_submission()


if __name__ == "__main__":
    main()
