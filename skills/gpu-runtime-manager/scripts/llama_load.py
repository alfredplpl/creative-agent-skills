#!/usr/bin/env python3
import argparse
import json

from gpu_runtime import GPUOwner, RuntimeManager, error_payload, load_config, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely acquire the GPU for llama.cpp")
    parser.add_argument("--config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(); manager = None
    try:
        cfg = load_config(args.config); setup_logging(cfg, args.json); manager = RuntimeManager(cfg)
        with manager.locked(): data = manager.acquire(GPUOwner.LLM, args.dry_run)
        if args.json: print(json.dumps({"ok": True, **data}, indent=2))
        elif args.dry_run:
            for action in data["would"]: print(f"would {action}")
        else: print("LLM ready")
        return 0
    except Exception as exc:
        payload = error_payload(exc, manager)
        print(json.dumps(payload, indent=2) if args.json else f"ERROR [{payload['error']}]: {payload['message']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
