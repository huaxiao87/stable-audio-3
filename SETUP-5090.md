# RTX 5090 (Blackwell / sm_120) 安装记录

这是在本机 RTX 5090 上跑 [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) 的环境记录与注意事项。代码本体就是官方 `stable-audio-3` 仓库,这里只记录针对 5090 做的改动。

## 已验证状态 ✅

```
torch        : 2.7.1+cu128
GPU          : NVIDIA GeForce RTX 5090
capability   : (12, 0)  => sm_120
FP16 matmul  : OK
stable_audio_3 import : OK
驱动 (UMD)   : CUDA 13.3 (向下兼容 cu128 wheel)
Python       : 3.11
包管理       : uv
```

## 针对 5090 做的唯一改动

`pyproject.toml` 里 PyTorch 的索引从默认的 **cu126 改成了 cu128**。
原因:RTX 5090 是 Blackwell **sm_120**,cu126 及以下的 wheel 没有 sm_120 内核,会报
`no kernel image is available` 或回退 CPU。cu128(CUDA 12.8)是第一个带 sm_120 的稳定 wheel,而 torch 2.7.1 又是 SA3 钉死的版本,正好都满足。

## 安装

```bash
cd /mnt/intel-nvme/stable-audio-lab

uv sync                              # 基础 (Python API + CLI)
uv sync --extra ui                   # + Gradio 界面
uv sync --extra lora                 # + LoRA 训练
uv sync --extra ui --extra lora      # 全部
```

## 模型一览

| 模型 | ID | 硬件 | 参数 | 最长 | 用途 |
|------|----|------|------|------|------|
| Small-Music | `small-music` | CPU | 433M | 120s | 纯音乐, 无需 GPU/flash-attn |
| Small-SFX | `small-sfx` | CPU | 433M | 120s | 音效, 无需 GPU/flash-attn |
| Medium | `medium` | GPU(CUDA) | 1.4B | 380s | 高质量, 需要 flash-attn 2 |
| Large | — | 仅 API | 2.7B | 380s | 不开源 |

> **你的 5090 主要用 `medium`**。`medium` 必须有正确的 flash-attn,否则输出是噪音(见下)。
> 想先跑通流程,可以先用 CPU 的 `small-music`(不需要 flash-attn)。

## ✅ flash-attn (medium 必需) —— 已搞定

已用系统 CUDA Toolkit 12.8 为 sm_120 从源码编译 `flash-attn==2.8.3` 并验证:
`flash_attn_func` 在 RTX 5090 上实跑成功。

```
flash-attn : 2.8.3 (sm_120 原生内核, 已验证可跑)
编译方式   : 系统 CUDA 12.8 (CUDA_HOME=/usr/local/cuda-12.8) + TORCH_CUDA_ARCH_LIST="12.0"
```

> ⚠️ flash-attn 不在 `pyproject.toml` / `uv.lock` 里。以后再跑 `uv sync` 会把它删掉!
> 要用 **`uv sync --inexact`** 才不会动 flash-attn。
> 若不慎删了, 重装命令:
> ```bash
> export CUDA_HOME=/usr/local/cuda-12.8 && export PATH="$CUDA_HOME/bin:$PATH"
> TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=24 \
>   uv pip install flash-attn==2.8.3 --no-build-isolation
> ```

---

## (历史记录) flash-attn 当初的坑

SA3 `medium` 依赖 Flash Attention 2 (`from flash_attn import flash_attn_func`)。
目前**没有**与 `cu128 + torch2.7 + cp311 + sm_120` 精确匹配的预编译 wheel:
- 官方/mjun0812 预编译 wheel 多为 cu126,不含 sm_120;
- 社区 sm_120 wheel (如 alkemiik-coder) 是 cu130/torch2.11,与 SA3 钉死的 torch 2.7.1 不匹配。

需要**从源码为 sm_120 编译**,这需要 `nvcc`(本机未装 CUDA toolkit)。
> 注意:pip 的 `nvidia-cuda-nvcc-cu12` 在 Linux 上只带 `ptxas`,**没有 `nvcc` 前端**,
> 因此无法仅靠 pip 在 venv 内完成编译。必须有真正的 CUDA toolkit。

### 方案 A(推荐, 需 sudo): 装系统 CUDA Toolkit 12.8 再编译

```bash
# 1) 安装 CUDA Toolkit 12.8 (Ubuntu 24.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8

# 2) 为 sm_120 编译 flash-attn (机器 384 核, 几分钟搞定)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
cd /mnt/intel-nvme/stable-audio-lab
uv pip install ninja packaging psutil
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 \
  uv pip install flash-attn==2.8.3 --no-build-isolation

# 3) 验证
uv run python -c "from flash_attn import flash_attn_func; import flash_attn; print('flash-attn', flash_attn.__version__)"
```

> 之后跑 `uv sync` 会因为 flash-attn 不在 lockfile 里而把它删掉,要用 `uv sync --inexact`。

### 方案 B(无需 sudo): 升级 torch 到 2.11+cu130 用现成 sm_120 wheel

代价是偏离 SA3 钉死的 2.7.1(需同时改 `pyproject.toml` 的 torch 版本与索引)。
你的驱动是 CUDA 13.3,cu130 反而是最贴合本机的。若选此路,告诉我,我来改并装。

## 快速开始 (先用 CPU 小模型跑通)

需要先在 Hugging Face 接受模型条款并登录:

```bash
uv run huggingface-cli login        # 或 export HF_TOKEN=...

# CPU 小模型, 不需要 flash-attn
uv run stable-audio --model small-music -p "lo-fi hip hop beat, 90 BPM" --duration 30 -o beat.wav

# GPU medium (需先装好 flash-attn)
uv run stable-audio --model medium -p "epic orchestral trailer, 120 BPM" --duration 60 -o trailer.wav

# Gradio 界面
uv sync --extra ui
uv run python run_gradio.py --model medium
```
