# 赛道一 · 动态链接预测（A 榜）

与 `warmup/` 隔离；原始数据通过目录联接只读引用 `../data_A`。

## 目录

```text
track1/
├── data/              # Junction → ../data_A（勿提交 git）
├── configs/           # 预处理与训练配置
├── scripts/           # stats、preprocess、基线启动
├── processed/         # 清洗/编码产物（gitignore）
├── src/               # 自研数据与模型代码
├── outputs/           # 提交用 *_result.csv
└── checkpoints/
```

## 数据联接

首次克隆后若 `data/` 不存在，在仓库根目录执行其一：

```powershell
powershell -ExecutionPolicy Bypass -File track1\scripts\link_data.ps1
```

或：

```powershell
cmd /c mklink /J "d:\cursor_file\计图\track1\data" "d:\cursor_file\计图\data_A"
```

## 常用命令

```powershell
$py = "D:\miniconda3\envs\jittorgeometric\python.exe"
$env:JITTOR_HOME = "D:\jittor_cache"

# 数据概览
& $py track1\scripts\stats.py

# 预处理（六步流水线；可选 pip install PyYAML 以编辑 default.yaml）
& $py -m pip install PyYAML
& $py track1\scripts\preprocess.py --dataset dataset1
& $py track1\scripts\preprocess.py --dataset dataset2

# 赛方基线（参考，勿改 craft_baseline/）
& $py craft_baseline\main.py --dataset dataset1 --data_dir track1\data --save_dir track1\checkpoints --output_dir track1\outputs
```

## 参考

- 赛方基线：`craft_baseline/`（只读）
- 模型实现：`JittorGeometric/jittor_geometric/nn/models/craft.py`
