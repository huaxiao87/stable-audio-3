#!/usr/bin/env python3
"""Phase 4.3: Inpaint / continuation — extend an existing clip."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUTPUTS_DIR, require_cuda, save_wav, timed_generate, write_run_meta

from stable_audio_3 import StableAudioModel

DEFAULT_PROMPT = "dreamy synth outro, fading reverb, gentle fade"


def load_or_generate_seed(
    model: StableAudioModel,
    seed_audio: Path | None,
    seed_duration: float,
    seed_prompt: str,
    seed: int,
) -> tuple[torch.Tensor, int]:
    sr = model.model_config["sample_rate"]
    if seed_audio is not None:
        wav, file_sr = torchaudio.load(str(seed_audio))
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        return wav, sr

    print(f"Generating seed clip ({seed_duration}s)...")
    audio = model.generate(
        prompt=seed_prompt,
        duration=seed_duration,
        steps=8,
        seed=seed,
    )
    if audio.dim() == 3:
        audio = audio[0]
    return audio, sr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small-music")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--seed-audio",
        type=Path,
        default=None,
        help="existing wav to continue; if omitted, generates a short seed first",
    )
    parser.add_argument("--seed-duration", type=float, default=30.0, help="seed length when auto-generating")
    parser.add_argument("--seed-prompt", default="lofi hip hop beat, warm vinyl, 85 BPM")
    parser.add_argument("--inpaint-start", type=float, default=None, help="seconds kept from seed (default: seed length)")
    parser.add_argument("--total-duration", type=float, default=60.0, help="output total length in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "03_inpaint")
    args = parser.parse_args()

    device = require_cuda(args.model)
    print(f"Loading {args.model} on {device}...")
    model = StableAudioModel.from_pretrained(args.model, device=device)

    seed_wav, sr = load_or_generate_seed(
        model,
        args.seed_audio,
        args.seed_duration,
        args.seed_prompt,
        args.seed,
    )

    seed_path = args.out_dir / "seed_input.wav"
    save_wav(seed_wav, seed_path, sr)

    seed_seconds = seed_wav.shape[-1] / sr
    inpaint_start = args.inpaint_start if args.inpaint_start is not None else seed_seconds
    inpaint_end = args.total_duration

    print(f"\nInpaint: keep [0, {inpaint_start}s), generate [{inpaint_start}, {inpaint_end}s)")
    audio, elapsed = timed_generate(
        model,
        prompt=args.prompt,
        duration=args.total_duration,
        steps=8,
        seed=args.seed,
        inpaint_audio=(sr, seed_wav),
        inpaint_mask_start_seconds=inpaint_start,
        inpaint_mask_end_seconds=inpaint_end,
    )

    out_path = args.out_dir / "continued.wav"
    save_wav(audio, out_path, sr)

    write_run_meta(
        args.out_dir / "run.json",
        experiment="03_inpaint",
        model=args.model,
        prompt=args.prompt,
        seed_audio=str(seed_path),
        inpaint_start_sec=inpaint_start,
        inpaint_end_sec=inpaint_end,
        total_duration_sec=args.total_duration,
        elapsed_sec=elapsed,
        output=str(out_path),
    )
    print(f"\nSeed:      {seed_path}")
    print(f"Output:    {out_path}  ({elapsed:.2f}s)")
    print(f"Metadata:  {args.out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
