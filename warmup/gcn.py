'''
热身赛：基于 GCN 的 Cora 节点分类任务
参赛选手需要填充注释为 TODO 的部分完成该赛题。
（在官方模板上填 TODO；本地无 GPU 时自动 spmm=False + NumPy 构图回退）
'''

import os.path as osp
import json
import pickle

import jittor as jt
from jittor import nn
import numpy as np
from jittor_geometric.nn import GCNConv
from jittor_geometric.ops import cootocsr, cootocsc
from jittor_geometric.nn.conv.gcn_conv import gcn_norm
from jittor_geometric.data import CSC, CSR

# ============================================================
# 基本配置
# ============================================================
if jt.has_cuda:
    jt.flags.use_cuda = 1
    _use_spmm = True
else:
    jt.flags.use_cuda = 0
    _use_spmm = False
jt.misc.set_global_seed(42)

# ============================================================
# 第一步：加载数据集
# ============================================================
data_path = osp.join('data', 'cora.pkl')

with open(data_path, 'rb') as f:
    raw = pickle.load(f)

# 将 numpy 数据转为 jittor 张量，构造 data 对象
class GraphData:
    pass

data = GraphData()
data.x = jt.array(raw['x'].astype(np.float32))
data.y = jt.array(raw['y'].astype(np.int64))
data.edge_index = jt.array(raw['edge_index'].astype(np.int64))
data.train_mask = jt.array(raw['train_mask'].astype(bool))
data.val_mask = jt.array(raw['val_mask'].astype(bool))
data.test_mask = jt.array(raw['test_mask'].astype(bool))
num_features = raw['num_features']
num_classes = raw['num_classes']

# 对特征做行归一化（等同于 T.NormalizeFeatures()）
row_sum = data.x.sum(dim=1, keepdims=True)
row_sum = jt.clamp(row_sum, min_v=1e-12)
data.x = data.x / row_sum


def _coo_to_csc_csr_numpy(edge_index, edge_weight, num_nodes):
    ei = edge_index.numpy()
    ew = np.asarray(edge_weight.numpy(), dtype=np.float32)
    src, dst = ei[0].astype(np.int64), ei[1].astype(np.int64)
    csc_order = np.argsort(dst, kind='stable')
    row_indices = src[csc_order].astype(np.int32)
    csc_w = ew[csc_order]
    col_off = np.zeros(num_nodes + 1, dtype=np.int32)
    np.add.at(col_off, dst + 1, 1)
    col_off = np.cumsum(col_off)
    csr_order = np.argsort(src, kind='stable')
    col_indices = dst[csr_order].astype(np.int32)
    csr_w = ew[csr_order]
    row_off = np.zeros(num_nodes + 1, dtype=np.int32)
    np.add.at(row_off, src + 1, 1)
    row_off = np.cumsum(row_off)
    return (
        CSC(jt.array(row_indices), jt.array(col_off), jt.array(csc_w)),
        CSR(jt.array(col_indices), jt.array(row_off), jt.array(csr_w)),
    )


# ============================================================
# 第二步：图的边归一化 + 稀疏格式转换
# ============================================================
v_num = data.x.shape[0]
edge_index, edge_weight = data.edge_index, None

edge_index, edge_weight = gcn_norm(
    edge_index, edge_weight, v_num,
    improved=False, add_self_loops=True
)

with jt.no_grad():
    try:
        data.csc = cootocsc(edge_index, edge_weight, v_num)
        data.csr = cootocsr(edge_index, edge_weight, v_num)
    except Exception:
        data.csc, data.csr = _coo_to_csc_csr_numpy(edge_index, edge_weight, v_num)

# ============================================================
# 第三步：定义 GCN 模型
# ============================================================
class GCNNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=256, dropout=0.8):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(num_features, hidden_dim, spmm=_use_spmm)
        self.conv2 = GCNConv(hidden_dim, num_classes, spmm=_use_spmm)

    def execute(self):
        x, csc, csr = data.x, data.csc, data.csr
        x = nn.relu(self.conv1(x, csc, csr))
        x = nn.dropout(x, self.dropout, is_train=self.is_training())
        x = self.conv2(x, csc, csr)
        return x

# 初始化模型和优化器
model = GCNNet(
    num_features=num_features,
    num_classes=num_classes,
    hidden_dim=256,
    dropout=0.8
)
optimizer = nn.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)

# ============================================================
# 第四步：定义训练函数
# ============================================================
def train():
    model.train()
    pred = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss = nn.cross_entropy_loss(pred, label)
    optimizer.step(loss)

# ============================================================
# 第五步：定义测试函数
# ============================================================
def test():
    model.eval()
    logits = model()
    accs = []

    for mask in [data.train_mask, data.val_mask]:
        pred, _ = jt.argmax(logits[mask], dim=1)
        acc = pred.equal(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    return accs

# ============================================================
# 第六步：训练模型
# ============================================================
best_val_acc = 0
best_state = None

for epoch in range(1, 201):
    train()
    train_acc, val_acc = test()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0:
        log = 'Epoch: {:03d}, Train Acc: {:.4f}, Val Acc: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc))

if best_state is not None:
    model.load_parameters(best_state)

print(f'\n最终结果: Val Acc: {best_val_acc:.4f}')

# ============================================================
# 第七步：生成并保存预测结果
# ============================================================
model.eval()

logits = model()
pred, _ = jt.argmax(logits, dim=1)

test_mask_np = data.test_mask.numpy().astype(bool)
test_indices = np.where(test_mask_np)[0]

result = {}
for idx in test_indices:
    result[str(int(idx))] = int(pred[int(idx)].item())

output_path = 'result.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"预测结果已保存到 {output_path}")
print(f"共预测 {len(result)} 个测试节点")
