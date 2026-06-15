# Stable Audio 3 推理架构

本文档描述 `stable-audio-lab` 仓库中 **推理（inference）** 的代码架构与数据流，便于理解从用户输入到输出音频的完整过程。

> 使用说明与 API 示例见 [inference.md](./workflows/inference.md)。

---

## 1. 总览

Stable Audio 3 的推理核心是一条 **latent 空间 Rectified Flow / RF Denoiser** 管线：

1. 文本与时长等条件经 **Conditioner** 编码
2. 可选音频经 **SAME 自编码器（AutoencoderPretransform）** 编码为 latent
3. **DiT（Diffusion Transformer）** 在 latent 空间做多步去噪
4. **SAME 解码器** 将 latent 还原为 44.1 kHz 波形

对外统一入口是 `StableAudioModel`（`stable_audio_3/model.py`），CLI 与 Gradio 都是对它的薄包装。

```mermaid
flowchart LR
    subgraph Input["输入"]
        P["文本 prompt"]
        D["时长 duration"]
        A["可选音频 init / inpaint"]
    end

    subgraph Latent["Latent 空间"]
        C["MultiConditioner"]
        N["随机噪声 / init 混合"]
        DiT["DiTWrapper → DiffusionTransformer"]
        N --> DiT
        C --> DiT
    end

    subgraph Audio["音频空间"]
        AE["SAME Autoencoder"]
        W["波形输出"]
    end

    P --> C
    D --> C
    A --> AE
    AE -->|encode| DiT
    DiT -->|denoised latent| AE
    AE -->|decode| W
```

---

## 2. 推理入口

```mermaid
flowchart TD
    subgraph Entry["入口层"]
        CLI["stable_audio_3/cli.py<br/>stable-audio 命令"]
        Gradio["run_gradio.py"]
        API["Python API<br/>StableAudioModel.generate()"]
        AEOnly["AutoencoderModel<br/>仅编解码"]
    end

    subgraph Core["核心门面"]
        SAM["StableAudioModel<br/>stable_audio_3/model.py"]
    end

    CLI -->|"from_pretrained + generate"| SAM
    Gradio -->|"create_diffusion_cond_ui()"| UI["interface/diffusion_cond.py<br/>generate_cond()"]
    UI --> SAM
    API --> SAM

    subgraph Opt["可选加速路径（独立实现）"]
        TRT["optimized/tensorRT/"]
        MLX["optimized/mlx/"]
    end
```

| 入口 | 文件 | 说明 |
|------|------|------|
| CLI | `stable_audio_3/cli.py` | `pyproject.toml` 注册为 `stable-audio` |
| Gradio Web UI | `run_gradio.py` → `interface/diffusion_cond.py` | 加载模型后调用 `generate_cond()` |
| Python API | `stable_audio_3/__init__.py` | 推荐：`StableAudioModel.from_pretrained()` |
| 独立 AE | `StableAudioModel` 同包的 `AutoencoderModel` | 不走扩散，仅 `encode()` / `decode()` |
| TensorRT | `optimized/tensorRT/scripts/` | CUDA 加速，逻辑镜像主路径 |
| MLX | `optimized/mlx/scripts/` | Apple Silicon 加速 |

---

## 3. 模型加载

```mermaid
sequenceDiagram
    participant User
    participant SAM as StableAudioModel
    participant MC as model_configs.all_models
    participant HF as HuggingFace Hub
    participant LU as loading_utils
    participant Factory as factory.py

    User->>SAM: from_pretrained("medium")
    SAM->>SAM: 选择 device: cuda → mps → cpu
    SAM->>MC: ModelConfig.resolve()
    MC->>HF: hf_hub_download(config + ckpt)
    HF-->>MC: 本地路径
    MC-->>SAM: model_config.json
    SAM->>LU: load_diffusion_cond(config, ckpt)
    LU->>Factory: create_diffusion_cond_from_config()
    Factory-->>LU: ConditionedDiffusionModelWrapper
    LU->>LU: load safetensors + .eval() + 可选 fp16
    LU-->>SAM: 完整模型
    SAM-->>User: StableAudioModel 实例
```

### 3.1 工厂组装（`factory.create_diffusion_cond_from_config`）

从 HuggingFace 下载的 `model_config.json` 驱动模型构建：

```mermaid
flowchart TD
    CFG["model_config.json"]

    CFG --> DiT["DiTWrapper<br/>→ DiffusionTransformer"]
    CFG --> COND["MultiConditioner"]
    CFG --> PT["AutoencoderPretransform"]
    CFG --> WRAP["ConditionedDiffusionModelWrapper"]

    COND --> T5["T5GemmaConditioner<br/>文本 → cross-attn"]
    COND --> NUM["NumberConditioner<br/>seconds_total → global"]

    PT --> SAME["AudioAutoencoder<br/>SAMEEncoder + SAMEDecoder"]
    PT --> PATCH["PatchedPretransform<br/>patch / unpatch"]

    DiT --> WRAP
    COND --> WRAP
    PT --> WRAP
```

### 3.2 已注册模型（`model_configs.py`）

| 模型 ID | HuggingFace Repo |
|---------|------------------|
| `small-music` / `small-sfx` | `stabilityai/stable-audio-3-small-*` |
| `medium` | `stabilityai/stable-audio-3-medium` |
| `*-base` | 对应 `*-base` 变体（需更高 `cfg_scale` 与更多 steps） |
| `same-s` / `same-l` | 独立 SAME 自编码器 |

---

## 4. 核心类关系

```mermaid
classDiagram
    class StableAudioModel {
        +from_pretrained()
        +generate()
        +load_lora()
        +set_lora_strength()
        -model: ConditionedDiffusionModelWrapper
        -same: AutoencoderPretransform
        -dit: DiTWrapper
    }

    class ConditionedDiffusionModelWrapper {
        +conditioner: MultiConditioner
        +model: DiTWrapper
        +pretransform: AutoencoderPretransform
        +get_conditioning_inputs()
        +sampling_dist_shift
        +diffusion_objective
    }

    class DiTWrapper {
        +model: DiffusionTransformer
        +forward(x, t, cfg_scale, ...)
    }

    class DiffusionTransformer {
        +ContinuousTransformer 层
        +CFG / APG / LoRA interval
    }

    class MultiConditioner {
        +forward(batch_metadata, device)
    }

    class T5GemmaConditioner {
        google/t5gemma-b-b-ul2
    }

    class NumberConditioner {
        seconds_total 嵌入
    }

    class AutoencoderPretransform {
        +encode()
        +decode(chunked)
        +downsampling_ratio
    }

    class AudioAutoencoder {
        +encode_audio()
        +decode_audio()
    }

    StableAudioModel --> ConditionedDiffusionModelWrapper
    ConditionedDiffusionModelWrapper --> DiTWrapper
    ConditionedDiffusionModelWrapper --> MultiConditioner
    ConditionedDiffusionModelWrapper --> AutoencoderPretransform
    DiTWrapper --> DiffusionTransformer
    MultiConditioner --> T5GemmaConditioner
    MultiConditioner --> NumberConditioner
    AutoencoderPretransform --> AudioAutoencoder
```

---

## 5. `generate()` 完整推理流水线

`StableAudioModel.generate()`（`model.py`）是推理的主编排函数。

```mermaid
flowchart TD
    START(["generate() 调用"])

    subgraph Prep["准备阶段"]
        BC["_build_conditioning_dicts()<br/>prompt + duration → dict"]
        AS["_adapt_sample_size()<br/>按 duration 对齐长度"]
        LS["audio_sample_size → latent_sample_size<br/>÷ downsampling_ratio"]
        MASK["构建 inpaint mask<br/>（秒 → 样本 → latent）"]
        SEED["torch.manual_seed(seed)<br/>torch.randn → noise"]
    end

    subgraph Cond["条件编码"]
        MC["model.conditioner()<br/>MultiConditioner"]
        GCI["get_conditioning_inputs()<br/>映射为 DiT 输入张量"]
        INP["注入 inpaint_mask<br/>inpaint_masked_input"]
    end

    subgraph AudioIn["可选音频输入"]
        INIT["init_audio → prepare_audio → encode"]
        INPAINT["inpaint_audio → encode"]
        MIX["init: noise = init*(1-σ) + noise*σ"]
    end

    subgraph Sample["sample_diffusion()"]
        SCHED["build_schedule() + dist_shift"]
        PAD["padding_mask（有效长度 + headroom）"]
        LOOP["采样循环 pingpong / euler / dpmpp / rk4"]
        DEC["pretransform.decode(chunked=?)"]
        ZERO["padding 区域清零"]
    end

    subgraph Post["后处理"]
        CLAMP["float32 + clamp(-1, 1)"]
        TRUNC["truncate_output_to_duration"]
        SAVE["torchaudio.save / Gradio WAV"]
    end

    START --> BC --> AS --> LS
    LS --> MASK --> SEED
    SEED --> MC --> INP --> GCI
    GCI --> INIT
    GCI --> INPAINT
    INIT --> MIX
    MIX --> SCHED
    SCHED --> PAD --> LOOP
    LOOP --> DEC --> ZERO --> CLAMP --> TRUNC --> SAVE
```

### 5.1 条件分两路

```mermaid
flowchart LR
    subgraph PathA["路径 A：经 Conditioner"]
        P["prompt"] --> T5["T5GemmaConditioner"]
        S["seconds_total"] --> NUM["NumberConditioner"]
        T5 --> CA["cross_attn_cond"]
        NUM --> GC["global_cond"]
    end

    subgraph PathB["路径 B：generate() 直接注入"]
        M["inpaint_mask"] --> IC["input_concat_cond"]
        MI["inpaint_masked_input"] --> IC
    end

    CA --> DiT["DiffusionTransformer"]
    GC --> DiT
    IC --> DiT
```

| 条件键 | 来源 | DiT 用途 |
|--------|------|----------|
| `prompt` | T5GemmaConditioner | Cross-attention |
| `seconds_total` | NumberConditioner | Global conditioning（控制生成长度） |
| `inpaint_mask` | `generate()` 内构建 | Channel concat（1=保留，0=重生成） |
| `inpaint_masked_input` | 编码后的 inpaint 音频 × mask | Channel concat |

---

## 6. 扩散采样（`inference/sampling.py`）

```mermaid
flowchart TD
    SD["sample_diffusion()"]

    SD --> OBJ{"diffusion_objective"}
    OBJ -->|"rf_denoiser（默认，8 步）"| PP["sampler: pingpong"]
    OBJ -->|"rectified_flow"| EU["sampler: euler"]

    SD --> BS["build_schedule(steps, sigma_max, dist_shift)"]
    BS --> DS["LogSNRShift 等<br/>distribution_shift.py"]

    PP --> LOOP["逐步去噪循环"]
    EU --> LOOP

    LOOP --> FWD["DiTWrapper.forward()<br/>x, t, cfg_scale, apg_scale, padding_mask"]

    FWD --> CFG["Classifier-Free Guidance<br/>batch_cfg + rescale_cfg"]
    CFG --> APG["Adaptive Projected Guidance<br/>apg_scale"]

    LOOP --> LAT["denoised latents"]
    LAT --> DEC{"decode?"}
    DEC -->|是| AE["pretransform.decode(chunked)"]
    DEC -->|否| OUT1["返回 latents"]
    AE --> MASK["× audio_mask 清零 padding"]
    MASK --> OUT2["返回波形 tensor"]
```

### 6.1 单步采样循环（概念）

```mermaid
sequenceDiagram
    participant S as Sampler (pingpong)
    participant M as DiTWrapper
    participant T as DiffusionTransformer

    Note over S: sigmas = [σ₀, σ₁, ..., 0]

    loop 每个 timestep t
        S->>M: forward(x, t, cond, cfg_scale, ...)
        M->>T: 拼接 cond/uncond batch（CFG）
        T-->>M: velocity / denoised prediction
        M-->>S: model output
        S->>S: RF 积分更新 x
    end

    S-->>S: 最终 latent x
```

### 6.2 关键默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `steps` | 8 | post-trained 模型蒸馏步数 |
| `cfg_scale` | 1.0 | post-trained 无需 CFG；`-base` 模型建议 ~7.0 |
| `sampler_type` | `pingpong`（rf_denoiser）/ `euler`（rectified_flow） | 自动按 objective 选择 |
| `init_noise_level` | 1.0 | A2A：越低越保留原音频 |
| `duration_padding_sec` | 6.0 | 有效长度外的 headroom |
| `chunked_decode` | `None` → 读 config | 分块解码降低 VRAM 峰值 |

---

## 7. 三种推理模式

### 7.1 Text-to-Audio（纯文本生成）

```mermaid
flowchart LR
    P["prompt"] --> C["conditioner"]
    D["duration"] --> C
    C --> N["纯随机噪声"]
    N --> DiT["DiT 去噪"]
    DiT --> AE["SAME decode"]
    AE --> W["波形"]
```

### 7.2 Audio-to-Audio（风格变换 / 变体）

```mermaid
flowchart LR
    A["init_audio"] --> PRE["prepare_audio<br/>重采样 + 声道对齐"]
    PRE --> ENC["SAME encode → latent"]
    ENC --> MIX["noise = init*(1-σ) + rand*σ"]
    P["prompt"] --> DiT
    MIX --> DiT["DiT 去噪"]
    DiT --> AE["decode"]
    AE --> W["波形"]
```

- `init_noise_level=1.0`：完全忽略 init，等同 T2A
- `init_noise_level` 越低：越接近原音频

### 7.3 Inpaint / Continuation（局部重生成 / 续写）

```mermaid
flowchart TD
    IA["inpaint_audio"] --> ENC["encode → latent"]
    SEC["start/end seconds"] --> MASK["构建 mask<br/>0=重生成, 1=保留"]
    ENC --> MI["inpaint_masked_input = audio × mask"]
    MASK --> CONCAT["input_concat_cond"]
    MI --> CONCAT
    CONCAT --> DiT["DiT 去噪"]
    DiT --> AE["decode"]
```

支持多个不连续区间（`inpaint_mask_start_seconds` / `end_seconds` 传 list）。

---

## 8. 预处理与后处理

```mermaid
flowchart TD
    subgraph Pre["预处理"]
        NA["numpy_audio_to_tensor()"]
        PA["prepare_audio()<br/>Resample + PadCrop + 声道"]
        ENC["AutoencoderPretransform.encode()"]
        INT["inpaint mask nearest 下采样到 latent"]
        TOK["T5Gemma tokenize + embed"]
    end

    subgraph Post["后处理"]
        Z["padding 区域 × audio_mask → 0"]
        F32[".to(float32).clamp(-1, 1)"]
        TR["按 duration 截断样本数"]
        CLI["cli._save_output → torchaudio.save"]
        GR["Gradio: int16 WAV + 可选 ffmpeg"]
    end

    NA --> PA --> ENC
    ENC --> DiT["扩散采样"]
    DiT --> Z --> F32 --> TR
    TR --> CLI
    TR --> GR
```

---

## 9. 设备与精度

```mermaid
flowchart TD
    FP["from_pretrained()"]
    FP --> DEV{"device?"}
    DEV -->|有 CUDA| CUDA["cuda + 默认 fp16"]
    DEV -->|有 MPS| MPS["mps, fp16=False"]
    DEV -->|否则| CPU["cpu, fp16=False"]

    CUDA --> WARN{"medium?"}
    WARN -->|无 GPU| PRINT["打印 CPU 警告"]

    GEN["generate()"]
    GEN --> DT["noise / cond 对齐 model_dtype"]
    GEN --> LAZY["T5Gemma 首次 forward 迁移到 device"]
    GEN --> CHUNK["chunked_decode 控制 VRAM"]
```

| 机制 | 位置 | 行为 |
|------|------|------|
| 自动选设备 | `StableAudioModel.from_pretrained()` | cuda → mps → cpu |
| FP16 | `loading_utils.load_diffusion_cond(model_half=True)` | 仅 CUDA；CPU/MPS 强制关闭 |
| TF32 关闭 | `StableAudioModel.__init__` | 数值稳定性 |
| Flash Attention | `medium` / SAME-L | 需安装 `flash-attn` |
| Gradio 清缓存 | `generate_cond()` | `torch.cuda.empty_cache()` |

---

## 10. LoRA（可选）

```mermaid
flowchart LR
    L["load_lora(ckpt_paths)"] --> LD["lora/loader.py<br/>load_and_apply_loras"]
    LD --> DiT["DiTWrapper"]
    LD --> COND["MultiConditioner"]
    S["set_lora_strength()"] --> STR["运行时调节强度"]
    GR["Gradio"] --> INT["按 sigma 区间启用不同 LoRA<br/>dit.py lora interval"]
```

---

## 11. 关键源文件索引

```
stable_audio_3/
├── model.py                    # StableAudioModel.generate() — 推理编排
├── cli.py                      # CLI 入口
├── factory.py                  # 从 config 组装模型
├── loading_utils.py            # safetensors 加载
├── model_configs.py            # HF 模型注册表
├── inference/
│   ├── sampling.py             # sample_diffusion() — 采样 + 解码
│   ├── audio_utils.py          # prepare_audio, numpy 转换
│   └── distribution_shift.py   # LogSNRShift 等 schedule 扭曲
├── models/
│   ├── diffusion.py            # ConditionedDiffusionModelWrapper, DiTWrapper
│   ├── dit.py                  # DiffusionTransformer + CFG/APG
│   ├── conditioners.py         # T5Gemma, Number, MultiConditioner
│   ├── autoencoders.py         # SAME 编解码
│   ├── pretransforms.py        # AutoencoderPretransform
│   └── lora/                   # LoRA 加载与强度
└── interface/
    └── diffusion_cond.py       # Gradio generate_cond()

run_gradio.py                   # Gradio 启动入口
optimized/tensorRT/             # CUDA TensorRT 加速
optimized/mlx/                  # Apple Silicon MLX 加速
```

---

## 12. 数据形状速查

以 `medium` 模型为例（概念值，具体以 `model_config.json` 为准）：

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入音频 | `(B, C, T_audio)` | 44.1 kHz，T ≈ duration × 44100 |
| Latent | `(B, io_channels, T_latent)` | T_latent = T_audio / downsampling_ratio |
| 噪声 | 同 latent | `torch.randn` 初始化 |
| 输出音频 | `(B, C, T_audio)` | decode 后 float32，范围 [-1, 1] |

---

## 13. 训练 vs 推理架构差异

> 依据 [Stable Audio 3 技术报告](https://arxiv.org/abs/2605.17991)（`2605.17991v1.pdf`）与 `stable_audio_3/training/` 源码整理。

训练和推理 **共用同一套神经网络骨架**（SAME + DiT + Conditioner），但训练是一条多阶段、带辅助模块的学习管线，推理是一条精简的前向采样管线。差异主要在目标函数、采样方式、数据流，以及只在训练时存在的组件。

### 13.1 一句话对照

| 维度 | 训练架构 | 推理架构 |
|------|----------|----------|
| 网络主体 | SAME（冻结）+ DiT + T5Gemma + 时长 / inpaint 条件 | **完全相同** |
| 独有组件 | 判别器、Teacher、CLAP-on-latent、OT 配对、随机 mask 增广 | **无**，只有采样循环 |
| 优化目标 | 速度场 MSE → 蒸馏 MSE → 对抗 + CLAP | **无 loss**，迭代去噪 |
| 采样 | 每步随机采一个 `t`，单次前向 | 固定 8 步 **ping-pong**（post-trained） |
| CFG | 训练时 dropout（p=0.1）学无条件分支 | post-trained **默认不用 CFG** |

### 13.2 训练三阶段管线

技术报告 Figure 10：扩散模型训练分三阶段，最终产出 post-trained checkpoint（`diffusion_objective = rf_denoiser`）。

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: Flow Matching（base）"]
        X0["真实 SAME latent x₀"]
        OT["Minibatch OT 噪声配对"]
        T["随机 t + 长度相关 shift"]
        MSE["MSE: 预测速度 v = ε − x₀"]
        X0 --> OT --> T --> MSE
    end

    subgraph Stage2["Stage 2: Distillation Warmup"]
        Teacher["冻结 Teacher<br/>15 步 DPM++ + CFG=5"]
        Student["Student 单步回归 x̂₀"]
        Teacher --> Student
    end

    subgraph Stage3["Stage 3: Adversarial Post-Training"]
        G["Generator（原 DiT）"]
        D["Discriminator（同架构 DiT + conv head）"]
        CLAP["CLAP-on-latent"]
        G --> D
        G --> CLAP
    end

    Stage1 --> Stage2 --> Stage3
    PT["post-trained checkpoint"]
    Stage3 --> PT
```

| 阶段 | 产物 | 推理时对应 checkpoint |
|------|------|----------------------|
| Flow Matching | `rectified_flow` base 模型 | `*-base`（多步 euler + 高 CFG） |
| Distillation Warmup | 单步 x̂₀ 能力（中间态） | 不单独发布 |
| Adversarial Post-Training | `rf_denoiser` post-trained | `small-music` / `medium` 等（8 步 pingpong） |

### 13.3 仅训练时存在的组件

| 组件 | 训练中的作用 | 推理时 |
|------|-------------|--------|
| **判别器 D** | 与 Generator 对抗，判断 latent 真假 | 不存在 |
| **Teacher 模型** | 蒸馏阶段生成 ODE 轨迹 | 不存在 |
| **CLAP loss** | 约束生成 latent 与 prompt 语义对齐 | 不存在（已 bake 进权重） |
| **OT coupling** | 批内重排噪声，拉直 flow 路径（`training/diffusion.py`） | 不存在 |
| **CFG dropout** | 10% 概率置零条件，学无条件分支 | post-trained 默认 `cfg_scale=1.0` |
| **随机 inpaint mask** | 80% 全生成 / 10% 随机段 / 10% 因果续写 | 用户指定 mask |
| **Silence augmentation** | 随机延长静音 latent（~4s 指数分布） | 固定 +6s padding，生成后截断 |
| **Masked loss** | padding 区不算 loss、attention mask | 只做 attention mask，不算 loss |

### 13.4 Base vs Post-trained 推理行为

| | **Base（Stage 1）** | **Post-trained（Stage 2+3）** |
|--|---------------------|-------------------------------|
| `diffusion_objective` | `rectified_flow` | `rf_denoiser` |
| 采样器 | `euler` 等 ODE 积分 | **`pingpong`** |
| 步数 | 50–100 步（技术报告） | **默认 8 步** |
| CFG | 需要（~7.0） | **不需要**（蒸馏时已内化 Teacher 的 CFG=5） |
| 模型学到什么 | 速度场 vθ，沿 ODE 多步积分 | 任意 t 下单步估计 x̂₀，再用 ping-pong 细化 |

Post-trained 推理核心（`inference/sampling.py`）：

```mermaid
flowchart TD
    Noise["纯噪声 ε（长度 ∝ duration + 6s）"]
    Loop["8 步 ping-pong 循环"]
    Denoise["单步去噪: x̂₀ = x − t·vθ(x,t,c)"]
    Renoise["重加噪: x = (1−t')x̂₀ + t'·ε'"]
    Decode["SAME decode（可 chunked）"]
    Trim["截断到请求时长"]

    Noise --> Loop
    Loop --> Denoise --> Renoise --> Loop
    Loop --> Decode --> Trim
```

对抗后训练让模型具备 **xₜ → x̂₀ 单步能力**，但 **ε → x̂₀ 一步仍困难**；ping-pong 将「一大步」拆成多步「去噪 → 重加噪」，且具备自纠错性（技术报告 §4, Figure 14）。

### 13.5 时间步调度（train–inference mismatch）

| | 训练 | 推理 |
|--|------|------|
| `t` 采样 | 截断 logit-normal，**按序列长度做 per-element shift**（μ=0.5~1.15） | **logSNR 均匀** 的固定 8 点 schedule，**与长度无关** |
| 目的 | 覆盖大量噪声水平，长序列偏更高噪声 | 8 步里每步放置更重要 |

技术报告明确这是 **有意的 mismatch**，实践中效果可接受（§4）。

### 13.6 变长处理

| | 训练 | 推理 |
|--|------|------|
| 序列长度 | batch 内 pad 到统一长度 | `L = ⌈(d + 6s) · fs / 4096⌉` |
| 有效内容 | `Leff = ⌈d·fs/4096⌉`，其余 silence latent | 同左，生成后按 `d` 截断 |
| Attention | padding 区 mask 掉 | 同样 mask，但不算 loss |
| Loss | 只在有效信号区（+ inpaint 分区） | 无 |

### 13.7 SAME 与 Inpainting

**SAME**

- **扩散训练**：SAME **冻结**，数据为 **离线预编码** latent；`training/diffusion.py` 中 `pre_encoded=True` 时跳过 encode。
- **推理**：A2A / inpaint 时 **在线** `encode()` 输入音频；最后 **在线** `decode()` 出波形。

SAME 自身另有独立训练（频谱重建、对抗、diffusion alignment、语义回归、对比对齐五类 loss），推理时不涉及。

**Inpainting**

- **训练**：`models/inpainting.py` 的 `random_inpaint_mask()`，loss 分「生成区（m=0）」与「保留区（m=1）」两项（报告 Eq.5）。
- **推理**：`generate()` 内按 `inpaint_mask_start/end_seconds` 构建 mask，逻辑相同（257 维 local-additive 条件），但 **不再随机**。

### 13.8 前向次数与 CFG 开销

| | 训练 | 推理（post-trained） |
|--|------|---------------------|
| 每训练/采样步 | 1 次前向（+ CFG dropout） | 8 步 × 1 次前向/步 |
| Teacher（蒸馏） | 15 步 × CFG 5.0 | — |
| CFG 双倍 batch | 仅 base 推理需要 | **省掉**（对消费级部署关键，报告 §4） |

### 13.9 与训练代码的边界（源码索引）

推理路径 **不经过** 以下训练专用逻辑，但复用相同底层模块：

| 文件 | 训练专用 | 推理复用 |
|------|----------|----------|
| `training/diffusion.py` | `DiffusionCondTrainingWrapper.training_step()` | 验证/demo 时调用 `sample_diffusion()` |
| `models/inpainting.py` | `random_inpaint_mask()` | — |
| `data/utils.py` | batch padding mask 构建 | `compute_effective_seq_len_from_conditioning()` 用于 schedule |
| `interface/diffusion_cond.py` | — | `is_rf_denoiser` 切换 UI 默认参数 |

---

*文档基于 `stable-audio-lab` 源码与技术报告梳理，主推理路径为 PyTorch `StableAudioModel.generate()`。*
