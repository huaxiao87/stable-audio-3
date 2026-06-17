# Stable Audio 3 复现进度表

对照 [学习路线图](../docs/learning-roadmap.md) 勾选。每完成一项，将 `[ ]` 改为 `[x]`。

---

## Phase 0：前置知识（3–5 天）

- [ ] 读过 [Model Overview](../docs/guides/model-overview.md) — latent diffusion 范式
- [ ] 理解 Flow Matching：`v = ε − x₀`，目标 `xt = (1-t)x₀ + tε`
- [ ] 理解 Classifier-Free Guidance：base 需 CFG≈7，post-trained 默认 CFG≈1
- [ ] 了解 AdaLN / cross-attn / local-additive 三种条件注入
- [ ] 了解 LoRA 基本原理（[LoRA Workflows](../docs/workflows/lora.md)）

---

## Phase 1：系统地图（1–2 天）

- [ ] 读过 [Inference Architecture](../docs/inference-architecture.md) §1–§5
- [ ] 读过 §13 训练 vs 推理差异
- [ ] 能画出三层结构：SAME（Layer A）→ DiT 三阶段（Layer B）→ Teacher/判别器（Layer C）
- [ ] 能说出三阶段各自产物：`*-base` / 中间态 / `small-music`·`medium` 等

---

## Phase 2：论文精读（约 1 周）

| 章节 | 完成 | 笔记文件 |
|------|------|----------|
| §2 Architecture + Fig.4–9 | [ ] | `notes/phase2/02-architecture.md` |
| §2.1 SAME + Fig.5–6 | [ ] | `notes/phase2/02.1-same.md` |
| §3.1 Variable-Length | [ ] | `notes/phase2/03.1-variable-length.md` |
| §3.2 Flow Matching | [ ] | `notes/phase2/03.2-flow-matching.md` |
| §3.3 Distillation | [ ] | `notes/phase2/03.3-distillation.md` |
| §3.4 Adversarial | [ ] | `notes/phase2/03.4-adversarial.md` |
| §4 Inference | [ ] | `notes/phase2/04-inference.md` |
| SAME 配套论文（可选） | [ ] | `notes/phase2/same-paper.md` |

每读完一节：回到 [Inference Architecture §13](../docs/inference-architecture.md#13-训练-vs-推理架构差异)，各写一句「训练时 / 推理时」总结。

---

## Phase 3：代码对照（约 1–2 周）

### 本 repo（按数据流）

- [ ] `stable_audio_3/model.py` → `generate()`
- [ ] `stable_audio_3/inference/sampling.py` — ping-pong / schedule
- [ ] `stable_audio_3/models/dit.py` — AdaLN / CFG / velocity
- [ ] `stable_audio_3/models/conditioners.py` — T5Gemma / duration
- [ ] `stable_audio_3/models/pretransforms.py` + `autoencoders.py`
- [ ] `stable_audio_3/training/diffusion.py` → `training_step()`
- [ ] `scripts/train_lora.py`

### 外部 repo（完整预训练）

- [ ] 克隆 [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)
- [ ] 找到训练配置 YAML / `model_config.json` 样例
- [ ] 了解 SAME 预编码数据管线
- [ ] 查找蒸馏与对抗 post-training 脚本（若已开源）

对照表：[code-trace/paper-to-code.md](./code-trace/paper-to-code.md)

---

## Phase 4：动手验证

- [ ] **4.1** `experiments/01_inference_base_vs_post.py` — base vs post-trained
- [ ] **4.2** `experiments/02_variable_length.py` — 变长生成
- [ ] **4.3** `experiments/03_inpaint.py` — inpaint / 续写
- [ ] **4.4** `experiments/04_same_encode_decode.py` — SAME 重建
- [ ] **4.5** LoRA 小数据集 500–1000 step（`../scripts/train_lora.py`）
- [ ] **4.6**（可选）读 loss 日志：`mse_signal` / `mse_context_loss`

---

## 里程碑

| 里程碑 | 完成 | 说明 |
|--------|------|------|
| M1：能解释端到端推理管线 | [ ] | Phase 1 + §4 论文 |
| M2：能对照代码讲清 Flow Matching loss | [ ] | Phase 2 §3.2 + `diffusion.py` |
| M3：跑通全部 Phase 4 实验 | [ ] | `experiments/` 四个脚本 |
| M4：完成一次 LoRA 微调 | [ ] | 小数据集验证 |
| M5：理解三阶段训练全貌 | [ ] | + stable-audio-tools 调研 |

---

*最后更新：创建复现工作区时初始化*
