# ICBHI2017 轻量四分类研究工程：CycleLite v0.1

本工程从零实现呼吸周期级四分类、预处理、轻量网络、训练、消融、蒸馏、评估和导出。目标是在可信的官方划分协议下探索 **ICBHI Score > 60%**，同时控制参数量与推理成本。

**当前状态：代码与合成数据流程已验证；没有训练真实 ICBHI 数据，没有预训练权重，没有可宣称的 60+ 成绩。** 合成检查的任何分数都不是实验结果。方法属于待实证的创新候选，不能直接写成论文已证实结论。

建议先阅读本文运行第一组 baseline/proposed，再阅读 `docs/RESEARCH_PLAN_zh.md`。实际验证记录在 `verification/`。

## 1. 任务和评分

这是异常呼吸音声学模式分类，四个类别不是四种疾病。正常音预测也不等同于排除疾病。

| 类别 ID | 英文 | 含义 | 标注 crackle | 标注 wheeze |
|---|---|---|---|---|
| 0 | normal | 无所标注的两类异常音 | 0 | 0 |
| 1 | crackle | 爆裂音/湿啰音 | 1 | 0 |
| 2 | wheeze | 哮鸣音 | 0 | 1 |
| 3 | both | 两类异常音并存 | 1 | 1 |

编码为 `label = crackle + 2 * wheeze`。混淆矩阵行是真值，列是预测。

\[
Sp=\frac{C_{00}}{N_0},\qquad
Se=\frac{C_{11}+C_{22}+C_{33}}{N_1+N_2+N_3},\qquad
Score=\frac{Sp+Se}{2}\times100.
\]

Crackle 判成 wheeze 必须扣分。不能把所有异常类合并后计算二分类灵敏度，也不能把四类 recall 的宏平均当成 ICBHI Score。此口径见 [CycleGuardian 的指标定义](https://arxiv.org/html/2502.00734v1)。

采用用户提供的**官方 train/test 文件**，保留官方测试集。默认再从官方训练患者中固定抽出约 20% 患者做开发验证集；所有实验使用同一份清单。完整标准数据的常用核对数为 920 条录音、126 位患者、6898 个周期，官方 train/test 周期数为 4142/2756，参考 [2025 年呼吸音蒸馏论文的数据说明](https://arxiv.org/html/2505.22027v1)。

本工程会检验数量、文件完整性及患者隔离，并保存划分文件 SHA-256；数量相同不能证明任意一个自制划分就是官方划分，仍需核实文件来源。官方资料入口：[ICBHI2017 数据库](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)。检索时官方主页读取超时，本包不虚构或附带一份猜测的官方清单。

## 2. 环境安装

推荐新建 Python 3.10–3.12 环境。实现不依赖 librosa、torchaudio 或 timm，可减少音频库与 CUDA 版本冲突。

```bash
conda create -n icbhi_lite python=3.10 -y
conda activate icbhi_lite
```

先按显卡驱动和目标运行环境，在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择适合的 CUDA wheel。若已有可用的 PyTorch 2.5–2.x GPU 环境，可直接使用。CPU 测试安装示例：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

在解压后的 `icbhi_lite` 项目根目录运行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

`verification/` 记录的测试环境为 PyTorch 2.8.0+cpu；CUDA/AMP 尚未在真实 GPU 上验证。requirements 是允许的版本范围，实际复现实验还应保存 `python -m pip freeze`、GPU 型号和驱动版本。

## 3. 数据预处理

准备一个含 `.wav` 与同名周期标注 `.txt` 的完整数据目录，以及 `ICBHI_challenge_train_test.txt`。支持递归搜索音频，不能指向含多份重复数据的父目录。划分文件支持“录音文件名 train/test”或“患者 ID train/test”两列，不猜测其他数字编码。

替换下面两个路径：

```bash
python -m icbhi_lite.prepare \
  --data-root /path/to/ICBHI_final_database \
  --official-split /path/to/ICBHI_challenge_train_test.txt \
  --config configs/proposed.yaml \
  --output cache/icbhi \
  --split-seed 42
```

完成后先读 `cache/icbhi/audit.json`。这里会列出 train/val/test 的患者数、周期数、类别分布、验证患者清单及文件摘要。出现患者交叉、缺录音、缺标注、重复文件、异常标签或计数错误时停止。允许不超过 20 ms 的边界越界修正，并记录被修正周期；超出范围需检查原始标注。缓存目录必须为空，失败后请检查原因并使用新目录重建。

`--allow-nonstandard-counts` 只为合成数据或子集开发提供，不应靠它掩盖真实数据缺失。它不会关闭患者隔离检查。

预处理过程：

1. PCM 转浮点，立体声均值转单声道，带抗混叠重采样到 16 kHz。
2. 去直流及二阶 50 Hz 高通，在完整录音上处理后按标注切出周期。该高通是离线零相位处理；当前版本不声明流式部署能力。
3. 两种 STFT：512 点（32 ms）和 2048 点（128 ms），统一 160 点（10 ms）步长。
4. 两种分辨率各提取 64 维 Mel 能量，频带 50–4000 Hz，生成 log-Mel 与固定参数 PCEN，共 4 通道。
5. 每个完整周期先计算特征，之后按 4 秒/50% 重叠划分窗口，保留末尾。短周期补零并记录有效长度。4 秒窗口为 401 个居中 STFT 帧。
6. 特征以 float16 缓存；训练用 float32 读取。只用当前训练清单计算每通道均值/标准差，推理严格复用。

没有强制拉伸周期，没有直接丢弃长周期尾部，没有强谱减降噪，也没有把周期标签当成每个子窗的精确标签。模型在全部窗口聚合以后接受一个周期级标签。重叠区域会获得多次上下文表示，默认未做覆盖次数重加权；可在论文中说明这一设计与限制。

如果原始录音采样率较低，重采样不会恢复原来不存在的高频信息。PCEN 是已有方法，本实现使用固定参数，不声称发明 PCEN 或实现了可训练 PCEN。[PCEN 原论文](https://arxiv.org/abs/1607.05666)

## 4. 先跑基线和完整模型

```bash
CUDA_VISIBLE_DEVICES=0 python -m icbhi_lite.train \
  --cache cache/icbhi --config configs/baseline.yaml \
  --output runs/baseline_seed1 --seed 1 --device cuda

CUDA_VISIBLE_DEVICES=0 python -m icbhi_lite.train \
  --cache cache/icbhi --config configs/proposed.yaml \
  --output runs/proposed_seed1 --seed 1 --device cuda
```

默认 150 epochs、batch size 16、AdamW、warmup+cosine、EMA、轻度 SpecAugment。验证每个 epoch 的 EMA 模型，按验证 Score 选最优；同分时按验证交叉熵选择。整个训练入口不评估官方测试集。

显存不足时用 `--batch-size 8` 或 `4`；CPU 排错用 `--device cpu --workers 0`。多窗口数会增加显存，不应只按 31 万参数判断训练开销。Windows 多进程有问题时也可设 workers=0。

每个 run 输出：

| 文件 | 用途 |
|---|---|
| `best.pt` | 验证选中的 EMA 模型及训练元数据 |
| `last.pt` | 最新训练状态，用于恢复 |
| `history.jsonl` | 每个 epoch 的损失分项、验证指标、耗时 |
| `best_validation.json` | 验证最好 epoch 与指标 |
| `validation_predictions.csv` | 最优模型对每个验证周期的概率 |
| `config.json` | 实际运行配置 |
| `normalization.json` | 训练统计量及拟合周期 ID |
| `run_summary.json` | 训练状态；标明未评估官方测试集 |

恢复需要相同配置、seed、数据与输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python -m icbhi_lite.train \
  --cache cache/icbhi --config configs/proposed.yaml \
  --output runs/proposed_seed1 --seed 1 --device cuda \
  --resume runs/proposed_seed1/last.pt
```

初次训练若使用过 `--epochs`、`--batch-size` 或 `--workers`，恢复时保持相同覆盖参数。为了保持学习率日程，不能直接修改总 epochs 后声称是严格续训。CPU 合成测试验证了中断恢复后权重与连续训练逐元素一致；跨硬件不承诺逐位一致。

## 5. 消融与多种子

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py \
  --cache cache/icbhi \
  --configs baseline ablation_frontend ablation_axis ablation_pooling proposed \
  --seeds 1 2 3 \
  --output-root runs/ablations --device cuda
```

脚本顺序训练，不会自动打开官方测试集。建议先跑 seed=1 的 baseline 和 proposed 验证趋势，再投入完整多种子实验。

| 配置 | 特征 | 主干 | 聚合 | 辅助损失 |
|---|---|---|---|---|
| baseline | 单分辨率 log-Mel | 3×3 深度可分离卷积 | 均值 | 无 |
| ablation_frontend | 双分辨率 log+PCEN | 同基线 | 均值 | 无 |
| ablation_axis | 同上 | 方向性深度卷积 | 均值 | 无 |
| ablation_pooling | 同上 | 同上 | 双事件注意力 | 无 |
| proposed | 同上 | 同上 | 双事件注意力 | 事件 BCE+边际一致性 |
| no_pcen | 双分辨率 log-Mel | 同完整模型 | 同完整模型 | 同完整模型 |
| no_consistency | 同完整模型 | 同完整模型 | 同完整模型 | 只有事件 BCE |
| tiny | 同完整模型 | 宽度 0.5 | 同完整模型 | 同完整模型 |
| teacher | 同完整模型 | 宽度 2.0 | 同完整模型 | 同完整模型 |
| distill | 宽度 1.0 完整模型 | 同完整模型 | 同完整模型 | 额外 KD |
| score_weight | 同完整模型 | 同完整模型 | 同完整模型 | 按 Score 的类别总数构造 CE 权重 |

这些配置是待检验因素，不代表每次添加模块必然提升分数。每种架构要分别报告实际参数和计算量；均值聚合基线保留相同分类 MLP 维度，事件查询在该配置不参与计算，但其参数仍包含在本包的全模型参数统计中。

## 6. 可选蒸馏

```bash
CUDA_VISIBLE_DEVICES=0 python -m icbhi_lite.train \
  --cache cache/icbhi --config configs/teacher.yaml \
  --output runs/teacher_seed1 --seed 1 --device cuda

CUDA_VISIBLE_DEVICES=0 python -m icbhi_lite.train \
  --cache cache/icbhi --config configs/distill.yaml \
  --teacher runs/teacher_seed1/best.pt \
  --output runs/distill_seed1 --seed 1 --device cuda
```

这里的教师是本工程宽度 2.0 的模型，**不含 AudioSet 预训练，也不等价于 AST 教师**。教师若没有更好或互补的验证性能，应取消蒸馏。本版要求相同训练周期、特征模式、归一化和分窗，防止教师接触学生验证患者。部署只用学生模型。

如果后续要接入已有 AST、CNN6 或 EfficientAT，需要适配各自前端、类别顺序和训练数据来源，不能把原始权重直接塞进这个接口。该外部教师适配尚未实现。

## 7. 锁定方案后再评估官方测试集

路线 A：报告开发验证集选择的模型，明确它只使用官方训练集的一部分拟合参数。

```bash
python -m icbhi_lite.evaluate \
  --cache cache/icbhi --checkpoint runs/proposed_seed1/best.pt \
  --split test --output runs/test_seed1 --device cuda
```

路线 B：在开发实验中先固定架构、超参数和 epochs，再重新初始化，在全部官方训练集重训。示例中的 80 **仅为命令示例**，必须换成开发验证确定的训练轮数。重训是从随机初始化开始，不是拿验证最优模型再继续训练。

```bash
CUDA_VISIBLE_DEVICES=0 python -m icbhi_lite.train \
  --cache cache/icbhi --config configs/proposed.yaml \
  --output runs/final_seed1 --seed 1 --device cuda --refit --epochs 80

python -m icbhi_lite.evaluate \
  --cache cache/icbhi --checkpoint runs/final_seed1/final.pt \
  --split test --output runs/final_test_seed1 --device cuda
```

重训重新拟合训练归一化统计量；不再使用开发验证集选 checkpoint，因为其患者已经参与训练；最终固定 epoch 的 EMA 模型保存为 `final.pt`。程序会拒绝把该模型对原验证患者的结果当成独立验证成绩。重训蒸馏时，教师也要在完整官方训练集上、以预先固定轮数另行训练。

对预先指定的 3–5 个 seeds 都评估，报告均值和样本标准差，不根据测试集挑 seed。训练集、重训协议、模型宽度、是否使用预训练/蒸馏都应如实注明。不得用最终测试表现继续挑阈值、epochs 或模块。

`metrics.json` 包含 Sp、Se、Score、accuracy、macro-F1、各类 recall、混淆矩阵、按患者聚类 bootstrap 的 Score 95% 区间和按设备分组指标。某设备没有正常/异常样本时 Score 为 null，避免虚构可比较分数。

```bash
python scripts/summarize_results.py \
  --results runs/final_test_seed1/metrics.json runs/final_test_seed2/metrics.json runs/final_test_seed3/metrics.json \
  --output runs/final_summary.json

python scripts/compare_predictions.py \
  --baseline runs/baseline_test/predictions.csv \
  --proposed runs/proposed_test/predictions.csv \
  --output runs/paired_difference.json
```

第二条命令要求两模型预测的是同一组周期，以患者为单位进行配对 bootstrap。它只描述给定两个模型的样本不确定性，不能替代多训练种子的方差分析。

## 8. 推理与轻量化测量

对完整录音按已有周期边界推理：

```bash
python -m icbhi_lite.infer \
  --checkpoint runs/proposed_seed1/best.pt \
  --wav /path/to/recording.wav \
  --annotation /path/to/recording.txt \
  --output predictions.json
```

只有音频本身已是一条完整呼吸周期时使用 `--single-cycle`，代替 `--annotation`。本版本不包含自动呼吸周期检测器，因此不能假设任意长录音就是一个周期。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m icbhi_lite.profile --config configs/proposed.yaml \
  --device cpu --threads 1 --repeats 50 --output profile.json

python -m icbhi_lite.export \
  --checkpoint runs/proposed_seed1/best.pt \
  --output exports/cyclelite.ts
```

主模型全参数量实测 **312,432（0.312M）**；tiny 为 **106,100（0.106M）**。主模型 FP32 参数本身约 1.19 MiB；`.pt` 完整训练 checkpoint 还包含优化器、EMA、原模型等，不能拿它的大小当纯部署权重大小。

profile 的 MACs 只统计 Conv/Linear；前端耗时单独统计，模型延迟不含磁盘 I/O、重采样及窗口组织。长周期多窗口成本会增加。CPU 结果只代表当前测量环境；真实部署需要重新测端到端延迟、内存和功耗。具体 JSON 见 `verification/`。

TorchScript 导出接受特征与有效长度，输出四分类 logits，自动检查不同 batch/window 数下的数值一致性；它不包含 SciPy 音频前端。该格式是兼容现有 PyTorch 环境的导出方式，本包不声称已完成手机端部署、ONNX 或 INT8 量化。

## 9. 工程验证与文件导航

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

smoke test 临时生成多采样率/立体声/多患者合成音频，验证预处理、患者划分、训练、恢复、蒸馏、推理、导出和泄漏拒绝。临时训练权重不会当作 ICBHI 模型交付。

| 文件 | 主要职责 |
|---|---|
| `icbhi_lite/prepare.py` | 官方清单解析、周期切分、审计、缓存 |
| `icbhi_lite/frontend.py` | 重采样、高通、双分辨率 Mel、PCEN |
| `icbhi_lite/data.py` | 训练统计量、全部窗口、有效长度、增强 |
| `icbhi_lite/model.py` | 轻量主干、方向性卷积、双事件聚合 |
| `icbhi_lite/losses.py` | 四分类、事件与一致性、蒸馏损失 |
| `icbhi_lite/train.py` | EMA、验证选择、恢复、完整训练集重训 |
| `icbhi_lite/metrics.py` | 正确的四分类官方指标与患者 bootstrap |
| `icbhi_lite/evaluate.py` | 冻结模型评估与逐周期预测输出 |
| `icbhi_lite/infer.py` | 原始 WAV 推理 |
| `icbhi_lite/profile.py` | 参数、计算量与延迟测量 |
| `icbhi_lite/export.py` | 数值核验后的模型导出 |
| `docs/RESEARCH_PLAN_zh.md` | 候选创新、消融矩阵和达标路线 |

获得真实数据后的第一批反馈应包含：`audit.json`、baseline/proposed 的 `best_validation.json`、`history.jsonl` 和实际训练环境。根据验证集错误类型调整后，再冻结正式实验方案。
