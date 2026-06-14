#!/usr/bin/env bash
# Stable Audio 3 medium —— 常用示例 (RTX 5090)
# 用法: 先看注释, 复制需要的命令单独跑; 或 `bash examples.sh basic`
set -euo pipefail
cd "$(dirname "$0")"

# 确保 uv 在 PATH (uv 装在 ~/.local/bin)
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

mkdir -p outputs

case "${1:-help}" in

  # ---- 0. 登录 (首次必须) -------------------------------------------------
  # medium 是门控模型: 先去 https://huggingface.co/stabilityai/stable-audio-3-medium
  # 点 "Agree and access" 接受条款, 再用 token 登录 (注意命令是 hf, 不是 huggingface-cli):
  login)
    uv run hf auth login
    ;;

  # ---- 1. 文本生成音乐 (最常用) ------------------------------------------
  basic)
    uv run stable-audio --model medium \
      -p "A triumphant UK bass-flavoured tech-house tune, pumping four-to-the-floor kick, syncopated 808 bass, gliding emotional synth leads, euphoric gospel house piano in the drop. 128 BPM" \
      --duration 60 --steps 8 --seed 42 \
      -o outputs/techhouse.wav
    ;;

  # ---- 2. 长曲 (medium 最长 380s) ----------------------------------------
  long)
    uv run stable-audio --model medium \
      -p "A cinematic orchestral epic, soaring strings, powerful brass, thunderous percussion, building to a heroic climax. 90 BPM" \
      --duration 240 --steps 8 \
      -o outputs/orchestral_4min.wav
    ;;

  # ---- 3. 一次出多条 (批量, 每条不同 prompt) -----------------------------
  batch)
    uv run stable-audio --model medium \
      -p "lofi hip hop beat, warm vinyl crackle, 85 BPM" \
         "energetic drum and bass, 174 BPM, rolling bassline" \
      --duration 30 \
      -o outputs/batch.wav
    ;;

  # ---- 4. 音效 / 一次性采样 (短时长) -------------------------------------
  sfx)
    uv run stable-audio --model medium \
      -p "TrackType: SFX, a blunt powerful wooden thud with low-mid body and slight analog distortion" \
      --duration 3 \
      -o outputs/thud.wav
    ;;

  # ---- 5. 音频转音频 (用一段音频做种子, 改风格) --------------------------
  #   init-noise-level 越高 = 越偏离原音频 (0.0-1.0)
  a2a)
    uv run stable-audio --model medium \
      -p "Heavy metal electric guitar" \
      --init-audio outputs/techhouse.wav --init-noise-level 0.6 \
      --duration 30 \
      -o outputs/restyled.wav
    ;;

  # ---- 6. 续写 / 延长一段音频 --------------------------------------------
  #   把 inpaint-start 设成原音频的长度, duration 设成想要的总长度
  continue)
    uv run stable-audio --model medium \
      -p "dreamy synth outro, fading reverb" \
      --inpaint-audio outputs/techhouse.wav \
      --inpaint-start 60 --inpaint-end 90 --duration 90 \
      -o outputs/extended.wav
    ;;

  # ---- 7. Gradio 网页界面 -------------------------------------------------
  ui)
    uv sync --extra ui --inexact   # --inexact: 别删掉 flash-attn
    uv run python run_gradio.py --model medium
    ;;

  *)
    echo "用法: bash examples.sh {login|basic|long|batch|sfx|a2a|continue|ui}"
    ;;
esac
