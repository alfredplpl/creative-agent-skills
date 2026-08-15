#!/usr/bin/env python3
import argparse
import json
from gpu_runtime import ConfigurationError, GPUMonitoringError, VRAMTimeout, load_config, wait_for_vram
def main():
    p = argparse.ArgumentParser(description="Wait with a finite timeout for measured free VRAM")
    p.add_argument("--gpu", type=int)
    p.add_argument("--free-mb", type=int, required=True)
    p.add_argument("--timeout", type=float)
    p.add_argument("--config")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        c = load_config(a.config)
        timeout = c["gpu"]["transition_timeout_seconds"] if a.timeout is None else a.timeout
        d = wait_for_vram(c["gpu"]["device"] if a.gpu is None else a.gpu,
                          a.free_mb, timeout, c["gpu"]["poll_interval_seconds"])
        print(json.dumps(d, indent=2) if a.json else f"VRAM ready: {d['free_vram_mb']} MB free")
        return 0
    except VRAMTimeout as e:
        print(json.dumps({"ok": False, "error": "vram_timeout", "message": str(e)}) if a.json else f"ERROR: {e}")
        return 1
    except ConfigurationError as e:
        print(json.dumps({"ok": False, "error": "configuration_error", "message": str(e)}) if a.json else f"ERROR: {e}")
        return 2
    except GPUMonitoringError as e:
        print(json.dumps({"ok": False, "error": "gpu_monitoring_unavailable", "message": str(e)}) if a.json else f"ERROR: {e}")
        return 3

if __name__ == "__main__":
    raise SystemExit(main())
