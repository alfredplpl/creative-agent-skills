#!/usr/bin/env python3
"""High-level CLI for safe single-GPU runtime transitions."""
import argparse
import json
from pathlib import Path

from gpu_runtime import (ConfigurationError, GPUOwner, RuntimeManager, error_payload,
                         load_config, setup_logging, validate_reference_image)
from h3_workflow import (REFERENCE_IMAGE_PLACEHOLDER,
                         build_h3_reference_workflow_from_prompt_file,
                         build_h3_workflow_from_prompt_file,
                         set_h3_reference_image)
from opencode_attachment import (cleanup_staged_attachment,
                                 resolve_opencode_attachment)


def show(data: dict, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps({"ok": True, **data}, indent=2))
        return
    if data.get("dry_run"):
        print(f"Dry run: target={data['target']} current={data['current']['gpu_owner']}")
        for action in data["would"]:
            print(f"would {action}")
        return
    gpu = data["gpu"]
    llama = data["llama"]
    comfy = data["comfyui"]
    llama_text = llama["model_state"] if llama["reachable"] else "unavailable"
    comfy_text = "idle" if comfy["idle"] else "busy" if comfy["reachable"] else "unavailable"
    print(f"GPU Owner: {data['gpu_owner'].upper()}")
    print(f"GPU: {gpu['used_vram_mb']} / {gpu['total_vram_mb']} MB used")
    print(f"llama.cpp: {llama_text}")
    print(f"ComfyUI: {comfy_text}")
    print(f"State: {data['detected_state'].upper()}")
    if data.get("atomic_video"):
        workflow = data["atomic_video"]
        print(f"Workflow: {workflow['status']} ({workflow['prompt_id']})")
        for item in workflow.get("files", []):
            path = "/".join(part for part in (item.get("subfolder"), item.get("filename")) if part)
            print(f"Output: {path}")
    if data.get("unexpected_processes"):
        print("Unexpected GPU processes:")
        for process in data["unexpected_processes"]:
            print(f"  PID {process['pid']} {process['name']} {process['used_vram_mb']} MB")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Safely switch one GPU between llama.cpp and ComfyUI")
    root.add_argument("--config", help="config.yaml path")
    commands = root.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true")
    for name in ("acquire", "release"):
        command = commands.add_parser(name)
        command.add_argument("runtime", choices=("llm", "video"))
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--json", action="store_true")
    switch = commands.add_parser("switch")
    switch.add_argument("source", choices=("llm", "video"))
    switch.add_argument("target", choices=("llm", "video"))
    switch.add_argument("--dry-run", action="store_true")
    switch.add_argument("--json", action="store_true")
    run_video = commands.add_parser(
        "run-video",
        help="atomically unload LLM, run one ComfyUI API workflow, and restore LLM",
    )
    source = run_video.add_mutually_exclusive_group(required=True)
    source.add_argument("--workflow", help="path to a ComfyUI API-format workflow JSON")
    source.add_argument("--prompt-file",
                        help="build the bundled MiniMax H3 workflow from this prompt")
    run_video.add_argument("--width", type=int, default=864)
    run_video.add_argument("--height", type=int, default=480)
    run_video.add_argument("--duration", type=float, default=3.0)
    run_video.add_argument("--seed", type=int)
    run_video.add_argument("--output-prefix", default="video/minimax_h3")
    reference = run_video.add_mutually_exclusive_group()
    reference.add_argument(
        "--reference-image",
        help="local character/style image; enables MiniMax H3 reference-to-video",
    )
    reference.add_argument(
        "--opencode-attachment", nargs="?", const="", metavar="FILENAME",
        help="use an OpenCode user image attachment (default: latest in active session)",
    )
    run_video.add_argument(
        "--reference-quality", choices=("match", "max"), default="match",
        help="match is faster; max preserves more reference detail",
    )
    run_video.add_argument("--opencode-session", help="disambiguate the OpenCode session")
    run_video.add_argument("--opencode-db", help="override the OpenCode database path")
    run_video.add_argument("--timeout", type=float,
                           help="workflow completion timeout in seconds")
    run_video.add_argument("--dry-run", action="store_true")
    run_video.add_argument("--json", action="store_true")
    return root


def load_workflow(path: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read workflow {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ConfigurationError("workflow must be a non-empty JSON object")
    return value


def main() -> int:
    args = parser().parse_args()
    json_mode = getattr(args, "json", False)
    manager = None
    staged_attachment = None
    try:
        cfg = load_config(args.config)
        setup_logging(cfg, json_mode)
        manager = RuntimeManager(cfg)
        if args.command == "status":
            result = manager.snapshot()
        elif args.command == "run-video":
            attachment_metadata = None
            reference_image = args.reference_image
            if args.opencode_attachment is not None:
                staged_attachment, attachment_metadata = resolve_opencode_attachment(
                    args.opencode_attachment or None,
                    session_id=args.opencode_session,
                    database=args.opencode_db,
                )
                reference_image = str(staged_attachment)
            elif args.opencode_session or args.opencode_db:
                raise ConfigurationError(
                    "--opencode-session and --opencode-db require --opencode-attachment"
                )
            if args.workflow and reference_image:
                raise ConfigurationError(
                    "reference images are supported with --prompt-file, not --workflow"
                )
            if args.reference_quality != "match" and not reference_image:
                raise ConfigurationError(
                    "--reference-quality requires a reference image"
                )
            if reference_image:
                validate_reference_image(reference_image)
            if args.workflow:
                workflow = load_workflow(args.workflow)
            elif reference_image:
                workflow = build_h3_reference_workflow_from_prompt_file(
                    args.prompt_file, REFERENCE_IMAGE_PLACEHOLDER,
                    reference_quality=args.reference_quality,
                    width=args.width, height=args.height, duration=args.duration,
                    seed=args.seed, output_prefix=args.output_prefix,
                )
            else:
                workflow = build_h3_workflow_from_prompt_file(
                    args.prompt_file, width=args.width, height=args.height,
                    duration=args.duration, seed=args.seed,
                    output_prefix=args.output_prefix,
                )
            with manager.locked():
                uploaded_image = None
                if reference_image and not args.dry_run:
                    uploaded_image = manager.upload_reference_image(reference_image)
                    set_h3_reference_image(workflow, uploaded_image)
                result = manager.run_video_workflow(workflow, args.dry_run, args.timeout)
                if reference_image:
                    if args.dry_run:
                        label = (f"OpenCode attachment {attachment_metadata['filename']}"
                                 if attachment_metadata else f"reference image {reference_image}")
                        result["would"].insert(
                            0, f"upload {label} to ComfyUI"
                        )
                    else:
                        result["atomic_video"]["reference_image"] = uploaded_image
                        if attachment_metadata:
                            result["atomic_video"]["opencode_attachment"] = attachment_metadata
        else:
            with manager.locked():
                if args.command == "acquire":
                    result = manager.acquire(GPUOwner(args.runtime), args.dry_run)
                elif args.command == "release":
                    result = manager.release(GPUOwner(args.runtime), args.dry_run)
                else:
                    if args.source == args.target:
                        raise ValueError("switch source and target must differ")
                    if args.dry_run:
                        first = manager.release(GPUOwner(args.source), True)
                        second = manager.acquire(GPUOwner(args.target), True)
                        result = {"dry_run": True, "target": args.target,
                                  "current": first["current"],
                                  "would": first["would"] + second["would"]}
                    else:
                        manager.release(GPUOwner(args.source))
                        result = manager.acquire(GPUOwner(args.target))
        show(result, json_mode)
        return 0
    except Exception as exc:
        payload = error_payload(exc, manager)
        if json_mode:
            print(json.dumps(payload, indent=2))
        else:
            print(f"ERROR [{payload['error']}]: {payload['message']}")
        return 1
    finally:
        if staged_attachment is not None:
            cleanup_staged_attachment(staged_attachment)


if __name__ == "__main__":
    raise SystemExit(main())
