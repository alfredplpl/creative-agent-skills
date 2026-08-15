#!/usr/bin/env python3
import argparse
import json

from gpu_runtime import comfy_status, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Show ComfyUI queue and execution status")
    parser.add_argument("--config")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        data = comfy_status(load_config(args.config))
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            state = "idle" if data["idle"] else "busy" if data["reachable"] else "unavailable"
            print(f"ComfyUI {data['version'] or 'unknown version'}: {state}")
            print(f"running={data['running_workflow_count']} pending={data['pending_workflow_count']}")
        return 0 if data["reachable"] else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}) if args.json else f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
