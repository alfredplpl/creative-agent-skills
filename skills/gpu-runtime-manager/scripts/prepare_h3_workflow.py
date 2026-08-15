#!/usr/bin/env python3
"""Create a MiniMax H3 ComfyUI API workflow without running it."""
import argparse
import json

from gpu_runtime import error_payload
from h3_workflow import build_h3_workflow_from_prompt_file, write_workflow


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Build a MiniMax H3 API workflow")
    command.add_argument("--prompt-file", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--width", type=int, default=864)
    command.add_argument("--height", type=int, default=480)
    command.add_argument("--duration", type=float, default=3.0)
    command.add_argument("--seed", type=int)
    command.add_argument("--output-prefix", default="video/minimax_h3")
    command.add_argument("--template")
    command.add_argument("--json", action="store_true")
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        options = {"width": args.width, "height": args.height, "duration": args.duration,
                   "seed": args.seed, "output_prefix": args.output_prefix}
        if args.template:
            options["template"] = args.template
        graph = build_h3_workflow_from_prompt_file(args.prompt_file, **options)
        write_workflow(args.output, graph)
        result = {"ok": True, "workflow": args.output, "width": args.width,
                  "height": args.height, "duration": args.duration,
                  "output_prefix": args.output_prefix}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Workflow: {args.output}")
        return 0
    except Exception as exc:
        payload = error_payload(exc)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"ERROR [{payload['error']}]: {payload['message']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
