# Phase 4 动手验证脚本

在**仓库根目录**运行：

```bash
uv run python repro/experiments/<script>.py [options]
```

输出默认写入 `repro/outputs/<实验名>/`。

## 前置条件

- 已 `uv sync`（LoRA 实验另加 `--extra lora`）
- HuggingFace 已登录：`uv run hf auth login`
- `medium` / `medium-base` 需要 CUDA

## 脚本

| 脚本 | 说明 |
|------|------|
| `01_inference_base_vs_post.py` | 同一 prompt 对比 post-trained 与 base |
| `02_variable_length.py` | 多时长生成，记录耗时 |
| `03_inpaint.py` | 续写 / inpaint |
| `04_same_encode_decode.py` | SAME 编解码重建误差 |

## 通用参数

多数脚本支持：

- `--model`：`small-music`（默认，CPU 可跑）/ `small-sfx` / `medium`
- `--seed`：随机种子
- `--out-dir`：输出目录（默认 `repro/outputs/...`）
