# jittor-Floutoble-track1

第六届计图（Jittor-7）人工智能挑战赛 · **Floutoble** 战队开源仓库  
维护者：Float246

- 热身赛：Cora 引文网络 GCN 节点分类（`warmup/`）
- 赛道一：图学习动态链接预测 A 榜（`track1/`）

基于 [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) 与 [JittorGeometric](https://github.com/AlgRUC/JittorGeometric)。

## 简介

本仓库包含计图挑战赛赛道一相关代码：数据预处理、赛方基线参考与自研实验代码。  
赛方原始 CSV **不包含在仓库中**，请按下方说明自行放置数据。

## 环境

- Python 3.10
- `jittor==1.3.7.16`
- JittorGeometric（本地 `pip install -e`）

详细步骤见 [docs/环境搭建.md](docs/环境搭建.md)。

```powershell
# 克隆 JittorGeometric（若尚未安装）
powershell -ExecutionPolicy Bypass -File scripts\clone_jittorgeometric.ps1

$env:JITTOR_HOME = "D:\jittor_cache"
D:\miniconda3\envs\jittorgeometric\python.exe -m pip install -e .\JittorGeometric
D:\miniconda3\envs\jittorgeometric\python.exe -m pip install -r track1\requirements.txt
```

## 数据

将头歌下发的 A 榜 CSV 放入 `data_A/`（目录结构见 `track1/README.md`），然后联接：

```powershell
powershell -ExecutionPolicy Bypass -File track1\scripts\link_data.ps1
```

## 训练（赛道一基线参考）

```powershell
$py = "D:\miniconda3\envs\jittorgeometric\python.exe"
$env:JITTOR_HOME = "D:\jittor_cache"

& $py craft_baseline\main.py --dataset dataset1 --data_dir track1\data --save_dir track1\checkpoints --output_dir track1\outputs
& $py craft_baseline\main.py --dataset dataset2 --data_dir track1\data --save_dir track1\checkpoints --output_dir track1\outputs
```

## 提交

打包 `track1/outputs/` 下生成的 CSV 为 `result.zip`（含 `dataset1.csv`、`dataset2.csv`），格式见 `craft_baseline/readme.md`。

## 目录

| 路径 | 说明 |
|------|------|
| `warmup/` | 热身赛 GCN |
| `track1/` | 赛道一工程 |
| `craft_baseline/` | 赛方基线（只读参考） |
| `docs/` | 环境与竞赛备忘 |

## 致谢

- 赛方基线：`craft_baseline/`
- 图学习框架：[JittorGeometric](https://github.com/AlgRUC/JittorGeometric)
