#!/usr/bin/env python
# Minimal CLI for ACEStepPipeline text-to-music (and optional audio2audio/edit extensions).
import argparse
import os
import sys
from typing import Optional

import torch

# Allow importing the pipeline from the repo subfolder.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ACE_SRC_PATH = os.path.join(REPO_ROOT, 'ace-step-main', 'src')
if ACE_SRC_PATH not in sys.path:
    sys.path.append(ACE_SRC_PATH)

from acestep.pipeline_ace_step import ACEStepPipeline  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Generate music using ACEStepPipeline")
    p.add_argument('--checkpoint_path', type=str, default=None, help='Directory containing downloaded checkpoints (optional).')
    p.add_argument('--prompt', type=str, required=True, help='Text prompt describing the music.')
    p.add_argument('--lyrics', type=str, default='', help='Optional lyrics (multi-line supported).')
    p.add_argument('--output_path', type=str, default='outputs', help='Directory or file path to save audio (defaults to directory).')
    p.add_argument('--duration', type=float, default=20.0, help='Target audio duration in seconds.')
    p.add_argument('--steps', type=int, default=60, help='Number of diffusion steps.')
    p.add_argument('--guidance_scale', type=float, default=15.0, help='Classifier-free guidance scale.')
    p.add_argument('--scheduler', type=str, default='euler', choices=['euler','heun','pingpong'], help='Scheduler type.')
    p.add_argument('--cfg_type', type=str, default='apg', choices=['apg','cfg'], help='Guidance algorithm.')
    p.add_argument('--omega_scale', type=float, default=10.0, help='Omega scale used in scheduler step.')
    p.add_argument('--seed', type=int, default=None, help='Random seed. If omitted, random.')
    p.add_argument('--format', type=str, default='wav', choices=['wav','ogg'], help='Output audio format.')
    p.add_argument('--device_id', type=int, default=0, help='CUDA device id.')
    p.add_argument('--bf16', action='store_true', help='Use bfloat16 dtype if supported.')
    p.add_argument('--float32', action='store_true', help='Force float32 dtype (overrides --bf16).')
    p.add_argument('--quantized', action='store_true', help='Load quantized checkpoints variant.')
    p.add_argument('--lora_path', type=str, default='none', help='Optional LoRA adapter name or local path.')
    p.add_argument('--lora_weight', type=float, default=1.0, help='LoRA weight scaling.')
    p.add_argument('--audio2audio_enable', action='store_true', help='Enable audio-to-audio (style transfer) mode.')
    p.add_argument('--ref_audio_input', type=str, default=None, help='Reference audio path for audio2audio.')
    p.add_argument('--ref_audio_strength', type=float, default=0.5, help='Strength of reference audio preservation (0-1).')
    p.add_argument('--share', action='store_true', help='(No effect here; retained for compatibility).')
    return p.parse_args()


def infer_dtype(args) -> torch.dtype:
    if args.float32:
        return torch.float32
    if args.bf16:
        # Fallback if bf16 unsupported on device.
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(args.device_id)
            if major >= 8:  # Ampere/Hopper & bfloat16 capable
                return torch.bfloat16
        return torch.float16 if torch.backends.mps.is_available() else torch.float32
    return torch.float32


def main():
    args = parse_args()

    save_path: Optional[str] = args.output_path
    # If user passed a directory ensure it exists.
    if save_path and (not os.path.splitext(save_path)[1]):  # no file extension, treat as directory
        os.makedirs(save_path, exist_ok=True)

    dtype = infer_dtype(args)

    pipeline = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        device_id=args.device_id,
        dtype='bfloat16' if dtype == torch.bfloat16 else 'float32',
        quantized=args.quantized,
    )

    manual_seeds = [args.seed] if args.seed is not None else None

    outputs = pipeline(
        format=args.format,
        audio_duration=args.duration,
        prompt=args.prompt,
        lyrics=args.lyrics,
        infer_step=args.steps,
        guidance_scale=args.guidance_scale,
        scheduler_type=args.scheduler,
        cfg_type=args.cfg_type,
        omega_scale=args.omega_scale,
        manual_seeds=manual_seeds,
        lora_name_or_path=args.lora_path,
        lora_weight=args.lora_weight,
        audio2audio_enable=args.audio2audio_enable,
        ref_audio_input=args.ref_audio_input,
        ref_audio_strength=args.ref_audio_strength,
        save_path=save_path,
        batch_size=1,
    )

    audio_paths = [p for p in outputs if isinstance(p, str) and p.endswith(f'.{args.format}')]
    meta = outputs[-1] if isinstance(outputs[-1], dict) else {}

    print('\nGeneration complete:')
    for pth in audio_paths:
        print('  Audio:', pth)
        json_path = pth.replace(f'.{args.format}', '_input_params.json')
        print('  Params JSON:', json_path)
    if 'timecosts' in meta:
        print('  Time Costs (s):', {k: f"{v:.2f}" for k, v in meta['timecosts'].items()})


if __name__ == '__main__':
    main()
