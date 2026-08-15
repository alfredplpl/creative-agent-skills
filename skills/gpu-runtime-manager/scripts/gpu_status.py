#!/usr/bin/env python3
import argparse
import json

from gpu_runtime import ConfigurationError, GPUMonitoringError, gpu_status, load_config

def main():
    p = argparse.ArgumentParser(description="Show the single GPU's measured state")
    p.add_argument("--gpu", type=int)
    p.add_argument("--config")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        cfg = load_config(a.config)
        data = gpu_status(cfg["gpu"]["device"] if a.gpu is None else a.gpu)
        if a.json: print(json.dumps(data, indent=2))
        else:
            print(f"GPU {data['gpu_index']}: {data['name']}")
            print(f"VRAM: {data['used_vram_mb']} / {data['total_vram_mb']} MB used ({data['free_vram_mb']} MB free)")
            print(f"Utilization: {data['utilization_percent']}%  Temperature: {data['temperature_c']} C")
            for x in data["processes"]:
                print(f"PID {x['pid']} {x['name']} {x['used_vram_mb']} MB")
        return 0
    except (ConfigurationError, GPUMonitoringError) as e:
        print(json.dumps({"ok": False, "error": str(e)}) if a.json else f"ERROR: {e}")
        return 3 if isinstance(e, GPUMonitoringError) else 2

if __name__ == "__main__":
    raise SystemExit(main())
