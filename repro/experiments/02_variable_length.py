#!/usr/bin/env python3
"""Phase 4.2: Variable-length generation — timing at multiple durations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUTPUTS_DIR, require_cuda, save_wav, timed_generate, write_run_meta

from stable_audio_3 import StableAudioModel

DEFAULT_PROMPT = "ambient electronic pad, slowly evolving textures, 80 BPM"
DEFAULT_DURATIONS = [10.0, 60.0, 120.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small-music")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=DEFAULT_DURATIONS,
        help="seconds to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "02_variable_length")
    args = parser.parse_args()

    device = require_cuda(args.model)
    print(f"Loading {args.model} on {device}...")
    model = StableAudioModel.from_pretrained(args.model, device=device)
    sr = model.model_config["sample_rate"]

    results = []
    for duration in args.durations:
        print(f"\n--- duration={duration}s ---")
        audio, elapsed = timed_generate(
            model,
            prompt=args.prompt,
            duration=duration,
            steps=8,
            seed=args.seed,
        )
        tag = f"{int(duration)}s"
        wav_path = args.out_dir / f"len_{tag}.wav"
        save_wav(audio, wav_path, sr)
        row = {"duration_sec": duration, "elapsed_sec": elapsed, "wav": str(wav_path)}
        results.append(row)
        print(f"  time={elapsed:.2f}s  ->  {wav_path}")

    write_run_meta(
        args.out_dir / "run.json",
        experiment="02_variable_length",
        model=args.model,
        prompt=args.prompt,
        seed=args.seed,
        results=results,
    )
    print(f"\nMetadata: {args.out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
