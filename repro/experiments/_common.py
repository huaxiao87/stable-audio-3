"""Shared helpers for repro/experiments scripts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPRO_ROOT / "outputs"


def detect_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_cuda(model_name: str) -> str:
    if model_name.startswith("medium") and not torch.cuda.is_available():
        raise SystemExit(f"{model_name} 需要 CUDA GPU，当前不可用。可改用 --model small-music。")
    return detect_device()


def paired_model_names(model: str) -> tuple[str, str]:
    """Return (post-trained, base) checkpoint names."""
    if model.endswith("-base"):
        base = model
        post = model.removesuffix("-base")
    else:
        post = model
        base = f"{model}-base"
    return post, base


def save_wav(audio: torch.Tensor, path: Path, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.dim() == 3:
        audio = audio[0]
    torchaudio.save(str(path), audio.cpu(), sample_rate)


def write_run_meta(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kwargs}
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def timed_generate(model, **kwargs):
    t0 = time.perf_counter()
    audio = model.generate(**kwargs)
    elapsed = time.perf_counter() - t0
    return audio, elapsed
