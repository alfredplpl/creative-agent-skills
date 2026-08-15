#!/usr/bin/env python3
"""Build a MiniMax H3 ComfyUI API workflow from the bundled template."""
from __future__ import annotations

import copy
import json
import os
import secrets
from pathlib import Path, PurePosixPath
from typing import Any

from gpu_runtime import ConfigurationError

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = SKILL_DIR / "assets" / "minimax-h3-t2v-api.json"
MAX_DURATION_SECONDS = 15.0
MAX_SEED = 2**64 - 1
REFERENCE_IMAGE_PLACEHOLDER = "gpu-runtime-manager/reference-image.png"
REFERENCE_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"


def _json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read {label} {source}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ConfigurationError(f"{label} must be a non-empty JSON object")
    return value


def read_prompt(path: str | Path) -> str:
    source = Path(path)
    try:
        prompt = source.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"cannot read prompt file {source}: {exc}") from exc
    if not prompt:
        raise ConfigurationError("prompt file must not be empty")
    return prompt


def _one_node(graph: dict[str, Any], class_type: str) -> dict[str, Any]:
    matches = [node for node in graph.values()
               if isinstance(node, dict) and node.get("class_type") == class_type]
    if len(matches) != 1:
        raise ConfigurationError(
            f"H3 template must contain exactly one {class_type} node; found {len(matches)}"
        )
    inputs = matches[0].get("inputs")
    if not isinstance(inputs, dict):
        raise ConfigurationError(f"{class_type} node has invalid inputs")
    return matches[0]


def _node_id(graph: dict[str, Any], class_type: str) -> str:
    matches = [node_id for node_id, node in graph.items()
               if isinstance(node, dict) and node.get("class_type") == class_type]
    if len(matches) != 1:
        raise ConfigurationError(
            f"H3 template must contain exactly one {class_type} node; found {len(matches)}"
        )
    return matches[0]


def _vae_node_id(graph: dict[str, Any], vae_name: str) -> str:
    matches = [node_id for node_id, node in graph.items()
               if isinstance(node, dict) and node.get("class_type") == "VAELoader"
               and node.get("inputs", {}).get("vae_name") == vae_name]
    if len(matches) != 1:
        raise ConfigurationError(
            f"H3 template must contain exactly one VAELoader for {vae_name}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{label} must be positive")
    return float(value)


def _dimension(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 32:
        raise ConfigurationError(f"{label} must be a positive multiple of 32")
    return value


def _output_prefix(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("output prefix must be a non-empty string")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError("output prefix must be a relative path without '..'")
    return path.as_posix()


def _comfy_input_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("reference image name must be a non-empty string")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError("reference image name must be relative without '..'")
    return path.as_posix()


def build_h3_workflow(prompt: str, *, width: int = 864, height: int = 480,
                      duration: float = 3.0, seed: int | None = None,
                      output_prefix: str = "video/minimax_h3",
                      template: str | Path = DEFAULT_TEMPLATE) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigurationError("prompt must be a non-empty string")
    width, height = _dimension(width, "width"), _dimension(height, "height")
    duration = _positive_number(duration, "duration")
    if duration > MAX_DURATION_SECONDS:
        raise ConfigurationError(f"duration must not exceed {MAX_DURATION_SECONDS:g} seconds")
    if seed is None:
        seed = secrets.randbits(64)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ConfigurationError(f"seed must be an integer between 0 and {MAX_SEED}")

    graph = copy.deepcopy(_json_object(template, "H3 template"))
    video = _one_node(graph, "MiniMaxH3ImageToVideo")["inputs"]
    video.update(prompt=prompt.strip(), width=width, height=height)
    _one_node(graph, "PrimitiveFloat")["inputs"]["value"] = duration
    noise = _one_node(graph, "RandomNoise")["inputs"]
    noise["noise_seed"] = seed
    noise.pop("value", None)
    _one_node(graph, "SaveVideo")["inputs"]["filename_prefix"] = _output_prefix(output_prefix)
    return graph


def build_h3_workflow_from_prompt_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return build_h3_workflow(read_prompt(path), **kwargs)


def build_h3_reference_workflow(
    prompt: str,
    reference_image: str,
    *,
    reference_quality: str = "match",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the installed MiniMax H3 single-image ref2va graph."""
    if "<Picture 1>" not in prompt:
        raise ConfigurationError("R2V prompt must reference the image as <Picture 1>")
    if reference_quality not in {"match", "max"}:
        raise ConfigurationError("reference quality must be match or max")

    graph = build_h3_workflow(prompt, **kwargs)
    video_id = _node_id(graph, "MiniMaxH3ImageToVideo")
    video = graph[video_id]
    video["class_type"] = "MiniMaxH3ReferenceToVideo"
    video.setdefault("_meta", {})["title"] = "MiniMax H3 Reference to Video"
    inputs = video["inputs"]
    inputs["audio_vae"] = [
        _vae_node_id(graph, "minimax_h3_audio_vae_fp32.safetensors"), 0
    ]
    reference_id = "gpu-runtime-manager:reference-image"
    inputs["ref_images.ref_image_0"] = [reference_id, 0]
    inputs["ref_image_size"] = reference_quality
    graph[reference_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": _comfy_input_name(reference_image)},
        "_meta": {"title": "Load Character Reference"},
    }
    _one_node(graph, "UNETLoader")["inputs"]["unet_name"] = REFERENCE_UNET
    return graph


def set_h3_reference_image(graph: dict[str, Any], reference_image: str) -> None:
    _one_node(graph, "LoadImage")["inputs"]["image"] = _comfy_input_name(reference_image)


def build_h3_reference_workflow_from_prompt_file(
    path: str | Path, reference_image: str, **kwargs: Any
) -> dict[str, Any]:
    return build_h3_reference_workflow(read_prompt(path), reference_image, **kwargs)


def write_workflow(path: str | Path, graph: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ConfigurationError(f"cannot write workflow {destination}: {exc}") from exc
