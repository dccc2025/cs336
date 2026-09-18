# Project Target：50M LLM from scratch（2× RTX 4080）

> 本文件是**当前阶段**的唯一规划源。SFT / DPO / RLHF **本期不做**，以后另开阶段再写。  
> 课程口径：体验 CS336 HW2 / HW3 里「怎么训、怎么看显存、怎么 debug」，不严格交作业。  
> 写代码前先读本文件；未经明示仍默认只在聊天框出代码。

---

## 1. 目标（锁定）

| 项 | 决定 |
|---|---|
| 模型 | **~50M** decoder-only LM（非 0.1B） |
| 硬件 | 2× RTX 4080（按你机子：每卡约 32GB，系统内存 64GB） |
| 数据 | FineWeb-Edu `sample-10BT`，已在数据盘 **~27–28GB** |
| 词表 | 在 FineWeb-Edu 上重训 **32k BPE**（不用 TinyStories 10k） |
| 上下文 | 目标 **2048**；冒烟可用 256/512 |
| 训练量 | Chinchilla 口径：50M × ~20 ≈ **~1B tokens**（不必吃满 10BT） |
| 系统件 | 已有 / 要接上：Triton kernels、FSDP、optimizer shard、fp16、activation checkpoint、FlashAttn |
| 交付 | **能 train + inference（续写）**；会做显存/耗时 profiling 与基础 debug |
| 明确取消（本期） | SFT、DPO、RLHF、学术问答 API、HADAR 语料 |

建议默认结构（训前用 `numel` 核对到 ~50M）：

```text
vocab=32000, seq=2048
d_model=512, num_layers=8, num_heads=8, d_ff=2048
tie_embeddings=true
```

成功标准（预训练）：valid loss 下降；续写不崩、像英文教育网页文本。

---

## 2. 数据现状

```text
/root/cs336/data  →  /root/autodl-tmp/cs336/data
/root/autodl-tmp/cs336/data/raw/fineweb/sample/10BT/*.parquet
≈ 27GB，14 个 parquet（FineWeb-Edu sample-10BT）✓ 已就绪
系统盘 overlay 仅 ~30GB → 大文件只放 autodl-tmp
```

后续数据产物（仍放数据盘，经 symlink）：

```text
data/clean/pretrain/       # 你手写清洗后的文本
data/tokenizer/            # 32k vocab + merges
data/tokenized/pretrain/   # memmap / shard，约 1B tokens 即可
```

---

## 3. 目录精简方案

**状态（2026-09-18）：仓库内清理已执行。** 已删 `modules/sft/`、`configs/**/sft.yaml`、`artifacts/checkpoints/{sft,dpo}/`、空壳 `modules/model/` 与 `modules/optim/`；`0.1b.yaml` → `configs/model/50m.yaml`；`pretrain.yaml` 目标改为 ~1B tokens。数据盘 FineWeb raw **未动**。`data/sft` 空目录若仍在 symlink 目标下可忽略或你稍后手动删。

### 3.1 当前树（train + infer + HW2/3）

```text
/root/cs336/
├── project_target.md          # 本规划（当前真相）
├── README.md                  # 短说明：怎么训 / 怎么续写
├── pyproject.toml
├── train_pretrain.py          # 2 卡入口（待写/待接）
├── generate.py                # inference 入口（待写）
├── main.py                    # 保留：TinyStories 单卡 HW1 冒烟（可标 deprecated）
│
├── configs/
│   ├── model/50m.yaml         # 新建；替换 0.1b 为默认
│   ├── tokenizer/bpe.yaml
│   ├── data/pretrain.yaml     # target_tokens ≈ 1e9
│   ├── train/pretrain_2gpu.yaml
│   └── profile.yaml
│
├── modules/
│   ├── module.py / bpe.py / tokenizer.py / loss.py / optimizer.py / scheduler.py / utils.py
│   ├── kernels/               # harness, vec, reduce, gemm, rope, flash_attn 保留
│   ├── parallel/              # fsdp, optimizer_shard 保留
│   ├── train/                 # loop, checkpoint 保留并接入口
│   ├── data/                  # clean / tokenize / pack（清洗你手写）
│   ├── infer/                 # generate
│   ├── profile/               # 显存 + 耗时（HW2/3 核心体验）
│   ├── model/                 # 可选：从 module 迁出；空则先删空壳或并回
│   └── optim/                 # 空壳则并回 optimizer.py，避免双份
│
├── scripts/
│   ├── train_pretrain.sh
│   ├── generate.sh
│   ├── profile_memory.sh
│   └── train_tinystories.sh   # 可选保留冒烟
│
├── tests/                     # kernel 数值对照
├── reports/                   # profiling 笔记（markdown / 截图说明）
├── data/ → autodl-tmp         # 大数据
└── artifacts/                 # ckpt / logs / profiles
    ├── checkpoints/pretrain/
    ├── logs/
    └── profiles/
```

### 3.2 已清理项

| 路径 | 动作 |
|---|---|
| `modules/sft/` | 已删 |
| `configs/data/sft.yaml`、`configs/train/sft.yaml` | 已删 |
| `artifacts/checkpoints/sft/`、`.../dpo/` | 已删 |
| `configs/model/0.1b.yaml` | 已换成 `50m.yaml` |
| 空壳 `modules/model/`、`modules/optim/` | 已删（逻辑仍在 `module.py` / `optimizer.py`） |

**不要删：** `modules/kernels/*`、`modules/parallel/*`、`modules/train/*`、HW1 核心、`data/raw`。

### 3.3 已有代码 vs 还缺

| 已有 | 还缺（按优先级） |
|---|---|
| HW1 模型 / AdamW / cosine | `configs/model/50m.yaml` |
| Triton L0–L4（Flash fwd + host bwd） | 清洗（你手写）、32k BPE、tokenize+pack≈1B |
| `parallel/fsdp.py`、`optimizer_shard.py` | 与 `TransformerLM` + FlashAttn **接线** |
| `train/loop.py`、`checkpoint.py` | 根目录 `train_pretrain.py` / `generate.py` 跑通 |
| FineWeb 28GB raw | `modules/profile/*` 真测 + `reports/` 笔记 |
| | activation checkpoint 接到 block；fp16 端到端 |

---

## 4. 阶段路线（只到预训练 + 续写）

```text
Phase A  仓库精简（按 §3 执行）+ 配置改为 50m / ~1B tokens
Phase B  数据：清洗 → 32k BPE → tokenize/pack（目标 ~1B tokens）
Phase C  单卡冒烟：50M + FlashAttn + amp + 短序列
Phase D  2 卡：FSDP + opt shard + activation checkpoint + train_pretrain
Phase E  HW2/HW3 体验：显存拆解、profiling、debug checklist（§5）
Phase F  正式训 ~1B tokens → generate 续写验收
───── 停在这里；SFT/DPO 以后再说 ─────
```

建议日历（粗估）：A 半天；B 1–3 天（BPE/tokenize 实现决定）；C–D 2–4 天；E 穿插 1–2 天；F 视吞吐约数小时到 1–2 天机器时间。

---

## 5. CS336 HW2 / HW3 体验 Guideline（显存 + debug）

目标不是交 PDF，而是**自己会量、会解释、会改**。

### 5.1 显存拆什么（每次实验记一张表）

对固定 `(micro_batch, seq, model)` 记录：

1. **参数** ≈ `4 × #params`（fp32）或 `2 × #params`（fp16 参数）  
2. **梯度** ≈ 与参数同量级  
3. **Adam 状态** ≈ `2 ×` 参数字节（m、v；FSDP/opt shard 后为 /world_size）  
4. **激活** ≈ 随 `batch × seq × layers × d` 涨；**2048 时通常是大头** → activation checkpoint 换时间  
5. **临时 / 碎片**（Flash 相对朴素 attn 省的是 `S²` 注意力矩阵）

命令习惯：

```text
nvidia-smi                    # 粗看每卡占用
torch.cuda.max_memory_allocated() / reserved()
# 对比开关：
#   - 有无 activation checkpoint
#   - 有无 FlashAttn（vs 朴素 S²）
#   - FSDP vs 单卡
#   - micro_batch = 1,2,4
```

写入 `reports/memory_50m.md`：每个配置一行 peak mem + tokens/s。

### 5.2 时间 / 瓶颈（HW 里的 profiling 思路）

1. 先 **端到端 step time**（含 sync）  
2. 再拆：**data loader / forward / backward / optimizer / 通信**  
3. 2 卡时看：是否几乎线性加速；若几乎不加速 → 通信或 CPU 瓶颈  
4. 工具可选：`torch.profiler`（chrome trace）、简单 `cuda.Event` 计时  

产物：`artifacts/profiles/` + `reports/profile_notes.md`。

### 5.3 Debug checklist（训练挂了按这个过）

| 症状 | 先查 |
|---|---|
| loss=NaN | lr 是否过大；fp16 是否需要跳过非有限 grad；softmax/Flash 数值 |
| loss 不降 | 数据是否全 padding；标签是否 shift；lr schedule；模型是否真在 train() |
| OOM | 降 micro_batch → 开 checkpoint → 确认 Flash → 查是否又物化了 S² |
| 2 卡挂 / 卡死 | `NCCL_DEBUG=INFO`；是否漏 `barrier`；checkpoint 是否只 rank0 写 |
| 续写乱码 | tokenizer 与训练是否同一套；`context_length`；采样温度 |
| 核结果不对 | `harness.ref_vs_triton`；先 S=128 再 2048 |

单卡正确 → 再开 FSDP；先短序 → 再 2048；先 TinyStories 冒烟 → 再 FineWeb。

### 5.4 建议最小实验矩阵（体验课感）

1. 50M，seq=512，batch=1，单卡，朴素 attn vs Flash → 记显存差  
2. 同上，开/关 activation checkpoint → 记显存与 step time  
3. 两卡 FSDP，同一 global batch → 记 tokens/s 与每卡显存  
4. seq=2048，找到刚好 OOM 的 batch，再退一档作为训练配置  

---

## 6. 近期执行顺序（仍先规划；你点名再写代码 / 再清目录）

1. ~~确认本文件；执行清理~~ **已完成（仓库内）**
2. 你写 **清洗**；同时可接 `train_pretrain.py` 用现有 TinyStories 做 2 卡冒烟
3. 32k BPE + tokenize ≈1B
4. 单卡 50M → 2 卡正式训
5. 按 §5 填 `reports/`
6. `generate.py` 验收续写

---

## 7. 与旧纲领的关系

| 旧锁定 | 本期 |
|---|---|
| 0.1B / ~2B tokens | **50M / ~1B tokens** |
| SFT + DPO | **推迟，不在本仓库当前路径** |
| 目录含 sft/ | **精简掉** |

原 skill 中的「总纲领原文」可作历史保留；**动手以本 `project_target.md` 为准**。
