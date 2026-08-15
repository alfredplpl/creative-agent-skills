#!/usr/bin/env python3
"""High-level CLI for safe single-GPU runtime transitions."""
import argparse
import json

from gpu_runtime import GPUOwner, RuntimeManager, error_payload, load_config, setup_logging


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
    return root


def main() -> int:
    args = parser().parse_args()
    json_mode = getattr(args, "json", False)
    manager = None
    try:
        cfg = load_config(args.config)
        setup_logging(cfg, json_mode)
        manager = RuntimeManager(cfg)
        if args.command == "status":
            result = manager.snapshot()
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


if __name__ == "__main__":
    raise SystemExit(main())
