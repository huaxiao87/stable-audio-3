# 论文 ↔ 代码对照表

技术报告：[arXiv:2605.17991](https://arxiv.org/abs/2605.17991) · 本地 PDF：`../../2605.17991v1.pdf`

按 [learning-roadmap Phase 3](../docs/learning-roadmap.md#phase-3代码对照约-1-2-周) 数据流顺序整理。读代码时在「我的笔记」列补充个人理解。

## 推理主线

| 顺序 | 源码 | 论文概念 | 我的笔记 |
|------|------|----------|----------|
| 1 | `stable_audio_3/model.py` → `generate()` | 推理编排、inpaint mask、条件注入 | |
| 2 | `stable_audio_3/inference/sampling.py` | ping-pong、euler、logSNR schedule、decode | |
| 3 | `stable_audio_3/models/dit.py` | AdaLN、CFG、velocity → x̂₀ | |
| 4 | `stable_audio_3/models/conditioners.py` | T5Gemma、duration 条件 | |
| 5 | `stable_audio_3/models/pretransforms.py` | SAME encode/decode 接口 | |
| 6 | `stable_audio_3/models/autoencoders.py` | SAME 架构实现 | |

## 训练（本 repo 可见部分）

| 代码位置 | 论文概念 | 我的笔记 |
|----------|----------|----------|
| `pre_encoded` / encode | 离线 latent | |
| `ot_coupling` | §3.2 Minibatch OT | |
| `dist_shift.shift(t, effective_seq_len)` | §3.1 长度相关 t shift | |
| `random_inpaint_mask()` | §3.2 inpaint 训练 | |
| `targets = noise - diffusion_input` | Flow matching 速度目标 `v = ε − x₀` | |
| `cfg_dropout_prob` | §3.5 CFG 训练 | |
| `training/diffusion.py` → `training_step()` | §3.1–3.2 训练核心 | |

## 本 repo 未覆盖（去 stable-audio-tools）

| 论文章节 | 预期位置 | 备注 |
|----------|----------|------|
| §3.3 Distillation | SAT 训练脚本 | Teacher 15 步 DPM++ + CFG=5 |
| §3.4 Adversarial | SAT 训练脚本 | 判别器 + CLAP-on-latent |
| §2.1 SAME 训练 | SAT / SAME repo | 五种 loss、TRB、4096× 压缩 |

## 训练 vs 推理（§13 速查）

读完每节后在此补充一句话：

| 模块 | 训练时 | 推理时 |
|------|--------|--------|
| 条件注入 | | |
| Inpaint | | |
| 采样 schedule | | |
| CFG | | |
| Ping-pong | | |
