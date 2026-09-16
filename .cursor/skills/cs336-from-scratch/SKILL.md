---
name: cs336-from-scratch
description: >-
  Long-horizon plan to train a 0.1B LLM from scratch on 2 GPUs: Triton,
  optimizer shard, FSDP, profiling, FlashAttention, FineWeb-Edu 2B tokens,
  SFT, DPO. Use whenever working in this repo, or when the user mentions
  CS336, 0.1B, Triton, FSDP, FlashAttention, FineWeb-Edu, 2B tokens, SFT,
  DPO, DeepSeek, HADAR, or writing kernels while downloading data.
---

# CS336 From Scratch（0.1B / 2 卡）

本 skill 是本仓库的长程总纲领。每次对话先读本文件，再动手。CS336 作业口径不严格执行；更关注从零训练完整过程。

## 总纲领（原文，不得改写）

就是 train 0.1B LLM from scratch, 用2卡。像cs336经历的事情一样

现在目录下已经写好对应的hw1的内容了

然后 我需要写triton, optimizaer shard, fsdp (2卡之间)，学会hw要求的那样profiling, analysis, flashattn, 显存优化

然后 我需要做hw4和hw5的内容。下载和制作数据集，清洗文档很网页tokenize then train LLM，制作成2B tokens用来训练。然后用SFT监督微调（学术风味的 用自己收集的cv llm vlm hadar-related文章 用api生成问答）然后用DPO做RLHF，完善其安全性和有用性。

所以需要先是我需要收集好预训练的数据和文本，然后下载的时候 我同时写算子的code，然后 训练，然后 SFT / RLHF

整个所有的过程未经我的明示，只准在聊天框里生成code

## 铁律：未经明示，代码只出现在聊天框

- 未经用户明确说「写入 / 落到仓库 / 改文件 / apply / 保存到磁盘」之前，**禁止**用 Write、StrReplace、EditNotebook、Delete，或任何会改仓库的手段。
- 允许：读仓库、搜索、回答、在聊天框贴完整可运行代码、解释取舍。
- 禁止：偷偷补脚本、改 `main.py` / `modules/`、加依赖、跑会写盘的下载/tokenize/训练。
- 用户只说「写一下 / 帮我写 / 看看怎么实现」= 只在聊天框给代码，不落盘。
- 本 skill、配套 rule、以及用户点名「整理目录」是已明示落盘的例外。
- 需要再落盘时，先列出将改的路径，等一句明确许可再写。

## 已锁定决定

| 项 | 值 |
|---|---|
| 布局 | 继续长 `modules/`，不另起包名 |
| GPU | RTX 4080 × 2，按用户声明每卡 32GB，内存 64GB |
| 预训练数据 | FineWeb-Edu 为主，英语，流式下载 |
| Tokenizer | 在 FineWeb-Edu 上重训 BPE，词表 32k |
| 模型 | GPT-2 small 像：768 / 12 / 12 / 3072 / seq 2048 |
| 训练量 | Chinchilla：约 0.1B 参数 × 约 2B tokens |
| 系统件 | Triton、opt shard、FSDP、手写 FlashAttn、profiling、显存优化都要；核心算子手写 |
| SFT | 学术风味；CV / LLM / VLM / HADAR（热成像/传感）；先几十篇跑通 |
| 问答 API | DeepSeek |
| DPO | 先不做；以后人工标一部分再 API 扩 |
| 成功标准 | 能续写、不崩、像英语/学术文本 |

超参以 `configs/` 为准，不要另写一套。

## 目录（已建，HW1 文件保持原地）

```
configs/{model,tokenizer,data,train}/   # yaml 已写入锁定值
modules/{model,kernels,parallel,optim,data,train,infer,profile,sft}/
scripts/                                # 现有 TinyStories 冒烟脚本
tests/  reports/
data/                                   # gitignore；流式 FineWeb-Edu 与 SFT 文章
artifacts/                              # gitignore；ckpt / profiles / logs
```

入口脚本 `train_pretrain.py` / `train_sft.py` / `generate.py` **尚未创建**。现有 `main.py` 与 `modules/*.py` 是 HW1 冒烟，不要挪走，直到用户明示迁移。

## 阶段

```
Phase 0  HW1 已完成（TinyStories 冒烟）
Phase 1  并行
         1a FineWeb-Edu 流式下载 → 清洗 →（稍后）32k BPE → 2B tokens
         1b 手写 Triton / FlashAttn、opt shard、2 卡 FSDP、profiling、显存
Phase 2  tokenize → 2 卡预训练 0.1B @ 2048
Phase 3  SFT：文章 → DeepSeek 问答 → 学术指令
Phase 4  DPO：人工标一部分 + API 扩
```

未做完 1a 口径和 1b 系统件，不要大规模预训练。未预训练完成，不做 SFT。未 SFT，不做 DPO。64GB 内存：数据必须流式，BPE 只能抽样训练。

## 各阶段要交付什么

### Phase 1a — 数据

流式拉 FineWeb-Edu，英语过滤，记来源与保留比例。产出可 tokenize 文本和数据卡。目标约 2B tokens。

### Phase 1b — 系统

核心算子手写（用户没写过 Triton，要完整走一遍）。先最小可运行 + 怎么测，再谈数字。可用 PyTorch 只做对照，不替代手写核。

### Phase 2 — 预训练

新 32k 词表 + 2B tokens，2 卡 FSDP 训 0.1B。记录 loss、吞吐、显存。成功 = 续写不崩、像英语/学术文本。

### Phase 3 — SFT

先几十篇证明流水线。域：CV、LLM、VLM、HADAR。像论文问答，不要闲聊。

### Phase 4 — DPO

SFT 之后再做。先人工偏好，再 API 扩。覆盖有用与安全。

## 长程写作

- 先对齐阶段和落盘许可，再写代码。一次只推当前阶段。
- 聊天框代码必须可独立粘贴，写清目标路径。
- 固定术语：0.1B、2 卡、2B tokens、FineWeb-Edu、Triton、optimizer shard、FSDP、FlashAttention、SFT、DPO、DeepSeek、HADAR。不要把 DPO 写成 PPO，不要把 2B tokens 写成 2B 参数。
- 不换模型家族、不加第三张卡、不把 SFT 做成通用助手，除非用户改纲领。

## 每轮回复怎么写

1. 点明阶段和落盘状态（默认：只出在聊天框）。
2. 需要代码就在聊天框给完整代码；默认不改仓库。
3. 结束时三行以内：刚完成什么、下一步、还缺哪句明示。
