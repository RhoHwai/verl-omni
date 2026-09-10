# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a deterministic FLUX/HPSv2 inference diagnostic without training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Any

_PACKAGE_NAMES = (
    "torch",
    "torch-npu",
    "transformers",
    "diffusers",
    "open-clip-torch",
    "verl",
    "verl-omni",
    "vllm",
    "vllm-omni",
)
_MODEL_COMPONENTS = ("scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "vae")
_WEIGHT_SUFFIXES = {".bin", ".pt", ".safetensors"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--hpsv2-pretrained-path", required=True, type=Path)
    parser.add_argument("--hpsv2-checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--reward-device", default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="Also score an identical PNG copied between the two servers.",
    )
    parser.add_argument(
        "--latents",
        type=Path,
        help="Reuse initial_latents.pt from another server instead of drawing new noise.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip FLUX loading/generation and only score --reference-image.",
    )
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Hash complete model files. By default large files use a first/last-chunk fingerprint.",
    )
    return parser.parse_args()


def _sha256(path: Path, *, full: bool, chunk_size: int = 4 * 1024 * 1024) -> tuple[str, str]:
    digest = hashlib.sha256()
    size = path.stat().st_size
    if full or size <= 2 * chunk_size:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest(), "full"

    with path.open("rb") as stream:
        digest.update(stream.read(chunk_size))
        stream.seek(-chunk_size, os.SEEK_END)
        digest.update(stream.read(chunk_size))
    digest.update(str(size).encode())
    return digest.hexdigest(), "first_last_4MiB_plus_size"


def _file_report(path: Path, *, full_hash: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path.absolute()),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
    }
    if path.is_symlink():
        report["symlink_target"] = os.readlink(path)
    if not path.exists():
        return report
    resolved = path.resolve()
    digest, digest_mode = _sha256(resolved, full=full_hash)
    report.update(
        {
            "resolved_path": str(resolved),
            "size": resolved.stat().st_size,
            "sha256": digest,
            "sha256_mode": digest_mode,
        }
    )
    return report


def _model_manifest(model_path: Path, *, full_hash: bool) -> list[dict[str, Any]]:
    paths = [model_path / "model_index.json"]
    for component in _MODEL_COMPONENTS:
        component_path = model_path / component
        if component_path.is_dir():
            paths.extend(path for path in component_path.iterdir() if path.is_file() or path.is_symlink())
    return [_file_report(path, full_hash=full_hash and path.suffix in _WEIGHT_SUFFIXES) for path in sorted(set(paths))]


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in _PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_report() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short"),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _extract_prompt(row: dict[str, Any]) -> tuple[str, str]:
    messages = _jsonable(row.get("prompt"))
    prompt = ""
    if isinstance(messages, list):
        user_contents = [
            item.get("content", "") for item in messages if isinstance(item, dict) and item.get("role") == "user"
        ]
        prompt = "".join(str(content) for content in user_contents)
    elif isinstance(messages, str):
        prompt = messages

    reward_model = _jsonable(row.get("reward_model"))
    ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
    return prompt, str(ground_truth)


def _read_parquet(path: Path, row_index: int, *, full_hash: bool) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Parquet contains no rows: {path}")
    if not 0 <= row_index < len(frame):
        raise IndexError(f"row-index {row_index} is outside [0, {len(frame)})")
    row = _jsonable(frame.iloc[row_index].to_dict())
    prompt, ground_truth = _extract_prompt(row)
    mismatch_row = _jsonable(frame.iloc[(row_index + 1) % len(frame)].to_dict())
    mismatch_prompt, mismatch_ground_truth = _extract_prompt(mismatch_row)
    mismatch = mismatch_ground_truth or mismatch_prompt
    file_info = _file_report(path, full_hash=full_hash)
    return (
        row,
        prompt,
        mismatch,
        {
            "file": file_info,
            "rows": len(frame),
            "columns": list(frame.columns),
            "row_index": row_index,
            "row_sha256": _canonical_digest(row),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "prompt_equals_ground_truth": prompt == ground_truth,
            "mismatch_prompt": mismatch,
        },
    )


def _tensor_report(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().float().cpu().contiguous()
    finite = torch.isfinite(value)
    digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
    report: dict[str, Any] = {
        "shape": list(tensor.shape),
        "source_dtype": str(tensor.dtype),
        "float32_sha256": digest,
        "finite_ratio": float(finite.float().mean().item()) if value.numel() else 1.0,
    }
    if value.numel():
        report.update(
            {
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "mean": float(value.mean().item()),
                "std": float(value.std().item()),
                "l2_norm": float(torch.linalg.vector_norm(value).item()),
            }
        )
    return report


def _image_report(image: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "size": list(image.size),
        "mode": image.mode,
        "uint8_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "channel_mean": [float(value) for value in array.mean(axis=(0, 1))],
        "channel_std": [float(value) for value in array.std(axis=(0, 1))],
    }


def _score_image(image: Any, prompt: str, mismatch_prompt: str, reward_device: str) -> dict[str, Any]:
    import torch

    from verl_omni.utils.reward_score.hpsv2_reward import _get_hpsv2_scorer, compute_score_hpsv2

    scorer = _get_hpsv2_scorer(reward_device)
    framework_score = compute_score_hpsv2(solution_image=image, ground_truth=prompt)
    mismatch_score = compute_score_hpsv2(solution_image=image, ground_truth=mismatch_prompt)
    empty_score = compute_score_hpsv2(solution_image=image, ground_truth="")

    image_input = scorer.preprocess(image.convert("RGB")).unsqueeze(0).to(scorer.device)
    text_input = scorer.tokenizer([prompt]).to(scorer.device)
    with torch.inference_mode():
        output = scorer.model(image_input, text_input)
        fp32_score = torch.diagonal(output["image_features"].float() @ output["text_features"].float().T)
    return {
        "framework": framework_score,
        "forced_fp32_dot": float(fp32_score.item()),
        "mismatch": mismatch_score,
        "empty": empty_score,
        "matched_minus_mismatch": float(framework_score["score"] - mismatch_score["score"]),
        "preprocessed_image": _tensor_report(image_input),
        "image_features": _tensor_report(output["image_features"]),
        "text_features": _tensor_report(output["text_features"]),
    }


def _load_device(device_name: str) -> Any:
    import torch

    if device_name.startswith("npu"):
        import torch_npu  # noqa: F401

    return torch.device(device_name)


def _run_flux(args: argparse.Namespace, prompt: str, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    from diffusers import FluxPipeline

    device = _load_device(args.device)
    dtype = getattr(torch, args.dtype)
    pipeline = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    clip_tokens = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    t5_tokens = pipeline.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=args.max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    with torch.inference_mode():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
        )

    first_forward: dict[str, Any] = {}
    original_forward = pipeline.transformer.forward

    def capture_first_forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
        hidden_states = forward_kwargs.get("hidden_states")
        result = original_forward(*forward_args, **forward_kwargs)
        if not first_forward:
            prediction = result[0] if isinstance(result, tuple) else result.sample
            if hidden_states is not None:
                initial_latents = hidden_states.detach().float().cpu()
                torch.save(initial_latents, output_dir / "initial_latents.pt")
                first_forward["initial_latents"] = _tensor_report(hidden_states)
            first_prediction = prediction.detach().float().cpu()
            torch.save(first_prediction, output_dir / "first_noise_prediction.pt")
            first_forward["noise_prediction"] = _tensor_report(prediction)
        return result

    pipeline.transformer.forward = capture_first_forward
    latents = torch.load(args.latents, map_location="cpu", weights_only=True) if args.latents else None
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.inference_mode():
        image = pipeline(
            prompt=prompt,
            prompt_2=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            max_sequence_length=args.max_sequence_length,
            generator=generator,
            latents=latents,
        ).images[0]
    pipeline.transformer.forward = original_forward
    image.save(output_dir / "generated.png")
    return image, {
        "backend": "diffusers.FluxPipeline",
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "clip_token_ids": _tensor_report(clip_tokens),
        "clip_non_padding_tokens": int((clip_tokens != pipeline.tokenizer.pad_token_id).sum().item()),
        "t5_token_ids": _tensor_report(t5_tokens),
        "t5_non_padding_tokens": int((t5_tokens != pipeline.tokenizer_2.pad_token_id).sum().item()),
        "prompt_embeds_t5": _tensor_report(prompt_embeds),
        "pooled_prompt_embeds_clip": _tensor_report(pooled_prompt_embeds),
        "text_ids": _tensor_report(text_ids),
        "first_forward": first_forward,
        "image": _image_report(image),
    }


def main() -> None:
    args = _parse_args()
    if args.score_only and args.reference_image is None:
        raise ValueError("--score-only requires --reference-image")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report: dict[str, Any] = {
        "status": "running",
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": _package_versions(),
            "git": _git_report(),
        },
    }

    try:
        for path in (args.model_path, args.parquet, args.hpsv2_pretrained_path, args.hpsv2_checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(path)
        os.environ["HPSV2_PRETRAINED_PATH"] = str(args.hpsv2_pretrained_path.resolve())
        os.environ["CUSTOM_REWARD_MODEL_PATH"] = str(args.hpsv2_checkpoint_path.resolve())
        os.environ["REWARD_DEVICE"] = args.reward_device

        _, prompt, mismatch_prompt, parquet_report = _read_parquet(
            args.parquet,
            args.row_index,
            full_hash=args.full_hash,
        )
        report["parquet"] = parquet_report
        report["files"] = {
            "model": _model_manifest(args.model_path, full_hash=args.full_hash),
            "hpsv2_pretrained": _file_report(args.hpsv2_pretrained_path, full_hash=args.full_hash),
            "hpsv2_checkpoint": _file_report(args.hpsv2_checkpoint_path, full_hash=args.full_hash),
        }

        images: dict[str, Any] = {}
        if args.reference_image is not None:
            from PIL import Image

            reference_image = Image.open(args.reference_image).convert("RGB")
            images["reference"] = reference_image
            report["reference_image"] = _image_report(reference_image)
        if not args.score_only:
            generated_image, generation_report = _run_flux(args, prompt, args.output_dir)
            images["generated"] = generated_image
            report["generation"] = generation_report

        report["reward"] = {
            name: _score_image(image, prompt, mismatch_prompt, args.reward_device) for name, image in images.items()
        }
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Diagnostic report: {report_path}")


if __name__ == "__main__":
    main()
