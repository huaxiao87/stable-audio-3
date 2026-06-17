# Stable Audio 3 论文独立复现工作区

本目录是 [Stable Audio 3 技术报告](https://arxiv.org/abs/2605.17991)（`arXiv:2605.17991`）的**个人复现与验证空间**，与主仓库 `stable_audio_3/` 源码分离，便于按 [学习路线图](../docs/learning-roadmap.md) 逐步推进。

## 与主仓库的关系

| 层级 | 位置 | 作用 |
|------|------|------|
| **推理 / LoRA 参考实现** | `../stable_audio_3/` | 官方开源代码，Phase 3–4 对照阅读 |
| **完整三阶段预训练** | [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | Flow Matching → 蒸馏 → 对抗后训练 |
| **本目录 `repro/`** | 此处 | 笔记、实验脚本、进度追踪、个人配置 |

论文 PDF：[`../2605.17991v1.pdf`](../2605.17991v1.pdf)

配套 SAME 论文：[arXiv:2605.18613](https://arxiv.org/abs/2605.18613)

## 目录结构

```
repro/
├── README.md              # 本文件
├── CHECKLIST.md           # 可勾选进度表（对应 roadmap 四阶段）
├── code-trace/            # 论文 ↔ 代码对照笔记
├── notes/                 # 精读笔记（按论文章节）
├── experiments/           # Phase 4 动手验证脚本
├── configs/               # 个人训练/实验配置（LoRA 等）
├── outputs/               # 实验输出音频与日志（gitignore）
└── data/                  # 本地数据集（gitignore）
```

## 快速开始

在仓库根目录执行（复用主项目 `.venv`）：

```bash
# 1. 确保环境已安装
cd /mnt/intel-nvme/stable-audio-lab
uv sync --extra lora

# 2. HuggingFace 登录（medium 等门控模型需要）
uv run hf auth login

# 3. 跑第一个复现实验：post-trained vs base 推理对比
uv run python repro/experiments/01_inference_base_vs_post.py --model small-music

# 4. 打开进度表，按阶段勾选
#    repro/CHECKLIST.md
```

有 CUDA GPU 时可将 `--model` 换成 `medium`，对比效果更明显。

## 推荐学习顺序

与 [learning-roadmap.md](../docs/learning-roadmap.md) 一致：

1. **Phase 0** — 扩散 / Flow Matching / CFG / AdaLN 前置概念
2. **Phase 1** — SAME + DiT + 三阶段训练系统地图
3. **Phase 2** — 按 §2 → §2.1 → §3.1–§4 拆读技术报告，笔记写入 `notes/phase2/`
4. **Phase 3** — 对照 `code-trace/paper-to-code.md` 读 `stable_audio_3/` 源码
5. **Phase 4** — 运行 `experiments/` 下脚本，验证推理行为

## 复现范围说明

| 目标 | 可行性 | 本目录支持 |
|------|--------|-----------|
| 理解架构 + 推理验证 | 高 | `experiments/` 脚本 |
| SAME 编解码 | 高 | `04_same_encode_decode.py` |
| Flow Matching 训练逻辑 | 中 | 对照 `training/diffusion.py` + 笔记 |
| LoRA 微调 | 高 | `configs/` + `../scripts/train_lora.py` |
| 从零 full pretrain | 极低 | 需 stable-audio-tools + 大规模算力 |

**不建议一开始就做的**：从零训练 medium、自行实现判别器 post-training——成本与工程复杂度极高（见 roadmap Phase 4 说明）。

## 实验脚本一览

| 脚本 | 对应 roadmap | 验证内容 |
|------|-------------|----------|
| `01_inference_base_vs_post.py` | Phase 4.1 | post-trained vs base：steps / cfg 差异 |
| `02_variable_length.py` | Phase 4.2 | 10s / 60s / 120s 变长生成 |
| `03_inpaint.py` | Phase 4.3 | 续写 / inpaint |
| `04_same_encode_decode.py` | Phase 4.4 | SAME latent 编解码重建 |

详细用法见 [experiments/README.md](./experiments/README.md)。
