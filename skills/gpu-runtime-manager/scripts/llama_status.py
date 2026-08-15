#!/usr/bin/env python3
import argparse
import json

from gpu_runtime import llama_status, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Show llama-server and configured model status")
    parser.add_argument("--config")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        data = llama_status(load_config(args.config))
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"llama.cpp: {'healthy' if data['healthy'] else 'unavailable'}")
            print(f"model: {data['model_identifier']} ({data['model_state']})")
            print(f"router mode: {data['router_mode']}")
        return 0 if data["reachable"] else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}) if args.json else f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
