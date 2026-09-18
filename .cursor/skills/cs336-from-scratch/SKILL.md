---
name: cs336-from-scratch
description: >-
  Train a ~50M LLM from scratch on 2 GPUs (FineWeb-Edu, Triton, FSDP,
  FlashAttention, profiling). Use when working in this repo or when the user
  mentions CS336, 50M, FineWeb-Edu, FSDP, FlashAttention, profiling, or
  project_target.md. SFT/DPO are deferred.
---

# CS336 From Scratch（当前：50M / 2 卡 / 仅预训练+推理）

**当前规划源：仓库根目录 [`project_target.md`](../../../project_target.md)。每次先读它。**

## 铁律：未经明示，代码只出现在聊天框

- 未经用户明确说「写入 / 落到仓库 / 改文件 / 保存到磁盘 / 执行清理」之前，禁止改仓库。
- 允许：读仓库、回答、在聊天框贴代码、解释。
- `project_target.md`、本 skill、配套 rule 在用户点名规划/清理时可落盘。

## 当前范围（2026-09 起）

| 做 | 不做（本期） |
|---|---|
| ~50M 预训练 + 续写 inference | SFT / DPO / RLHF |
| FineWeb-Edu（已有 ~28GB sample-10BT）→ 清洗 → 32k BPE → ~1B tokens | 学术问答 API、HADAR 语料 |
| Triton / FlashAttn / FSDP / opt shard / fp16 / activation checkpoint | 严格交 CS336 PDF |
| HW2/HW3 式显存与 profiling / debug（见 project_target.md §5） | |

硬件：2×4080；数据在 `/root/autodl-tmp/cs336/data`（经 `data/` symlink）。

## 阶段

```text
A 目录精简（去掉 sft/dpo）→ B 数据 → C 单卡冒烟 → D 2 卡训
→ E profiling/debug → F 续写验收 → 停；SFT 以后再说
```

固定术语：50M、~1B tokens、FineWeb-Edu、FSDP、FlashAttention。不要把本期做成 SFT 助手。
