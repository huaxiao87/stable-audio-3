#!/usr/bin/env python3
"""Phase 4.4: SAME encode/decode round-trip reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OUTPUTS_DIR, detect_device, save_wav, write_run_meta

from stable_audio_3 import AutoencoderModel, StableAudioModel

AE_FOR_MODEL = {
    "small-music": "same-s",
    "small-sfx": "same-s",
    "medium": "same-l",
}


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def reconstruction_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    n = min(original.shape[-1], reconstructed.shape[-1])
    a = original[..., :n].float()
    b = reconstructed[..., :n].float()
    return float(torch.mean((a - b) ** 2).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae", default=None, help="same-s or same-l (inferred from --model if set)")
    parser.add_argument("--model", default="small-music", help="used to pick AE and optional seed generation")
    parser.add_argument("--audio", type=Path, default=None, help="input wav; auto-generated if omitted")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "04_same_encode_decode")
    args = parser.parse_args()

    ae_name = args.ae or AE_FOR_MODEL.get(args.model, "same-s")
    device = detect_device(prefer_cuda=ae_name == "same-l")

    print(f"Loading AutoencoderModel '{ae_name}' on {device}...")
    ae = AutoencoderModel.from_pretrained(ae_name, device=device)
    sr = ae.sample_rate

    if args.audio is not None:
        wav = load_audio(args.audio, sr)
        source = "file"
    else:
        print(f"Generating {args.duration}s test audio via {args.model}...")
        sa3 = StableAudioModel.from_pretrained(args.model, device=device)
        out = sa3.generate(
            prompt="acoustic guitar fingerpicking, warm tone",
            duration=args.duration,
            steps=8,
            seed=args.seed,
        )
        wav = out[0] if out.dim() == 3 else out
        source = "generated"
        del sa3

    orig_path = args.out_dir / "original.wav"
    save_wav(wav, orig_path, sr)

    # encode → decode
    with torch.inference_mode():
        latent = ae.encode(wav, sr)
        recon = ae.decode(latent)
    if recon.dim() == 3:
        recon = recon[0]

    recon_path = args.out_dir / "reconstructed.wav"
    save_wav(recon.cpu(), recon_path, sr)

    mse = reconstruction_mse(wav, recon.cpu())
    latent_shape = tuple(latent.shape)

    write_run_meta(
        args.out_dir / "run.json",
        experiment="04_same_encode_decode",
        ae=ae_name,
        source=source,
        sample_rate=sr,
        latent_shape=latent_shape,
        mse=mse,
        original=str(orig_path),
        reconstructed=str(recon_path),
        note="latent 维度与压缩比见论文 §2.1 SAME",
    )

    print(f"\nLatent shape: {latent_shape}")
    print(f"MSE (waveform): {mse:.6f}")
    print(f"Original:       {orig_path}")
    print(f"Reconstructed:  {recon_path}")
    print(f"Metadata:       {args.out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
