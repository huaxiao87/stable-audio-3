#!/usr/bin/env python3
"""Phase 4.1: Compare post-trained vs base inference (steps / cfg)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing repro as a package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    OUTPUTS_DIR,
    paired_model_names,
    require_cuda,
    save_wav,
    timed_generate,
    write_run_meta,
)

from stable_audio_3 import StableAudioModel

DEFAULT_PROMPT = (
    "trap drums, hip hop beat, punchy kick, crisp hi-hats, 120 BPM"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small-music", help="post-trained model name")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "01_base_vs_post")
    args = parser.parse_args()

    post_name, base_name = paired_model_names(args.model)
    device = require_cuda(post_name)

    post_cfg = {"steps": 8, "cfg_scale": 1.0}
    base_cfg = {"steps": 50, "cfg_scale": 7.0}

    out_dir = args.out_dir
    results = []

    for label, name, gen_cfg in [
        ("post", post_name, post_cfg),
        ("base", base_name, base_cfg),
    ]:
        print(f"\n=== Loading {name} ({label}) ===")
        model = StableAudioModel.from_pretrained(name, device=device)
        sr = model.model_config["sample_rate"]

        audio, elapsed = timed_generate(
            model,
            prompt=args.prompt,
            duration=args.duration,
            seed=args.seed,
            **gen_cfg,
        )

        wav_path = out_dir / f"{label}_{name.replace('-', '_')}.wav"
        save_wav(audio, wav_path, sr)
        row = {
            "label": label,
            "model": name,
            "seconds": elapsed,
            "wav": str(wav_path),
            **gen_cfg,
        }
        results.append(row)
        print(f"  steps={gen_cfg['steps']} cfg={gen_cfg['cfg_scale']} time={elapsed:.2f}s")
        print(f"  saved: {wav_path}")

        del model

    write_run_meta(
        out_dir / "run.json",
        experiment="01_inference_base_vs_post",
        prompt=args.prompt,
        duration=args.duration,
        seed=args.seed,
        results=results,
        note="post-trained: 8 steps + cfg≈1; base: 50 steps + cfg≈7 (见 inference-architecture.md)",
    )
    print(f"\nMetadata: {out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
