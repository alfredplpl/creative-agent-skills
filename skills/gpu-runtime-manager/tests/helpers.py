import copy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from gpu_runtime import RuntimeManagerError  # noqa: E402


def config(temp_dir: str) -> dict:
    return {
        "gpu": {"device": 0, "poll_interval_seconds": 0.001,
                "transition_timeout_seconds": 0.03, "owner_process_threshold_mb": 1024},
        "process_detection": {"llm_patterns": ["llama-server"], "video_patterns": ["comfyui"]},
        "llm": {"runtime": "llama.cpp", "base_url": "http://llama", "model": {"name": "qwen"},
                "required_free_vram_mb": 18000, "management_mode": "router",
                "startup_timeout_seconds": 0.03, "managed_process": {"command": []}},
        "video": {"runtime": "comfyui", "base_url": "http://comfy",
                  "model": {"name": "minimax-h3"}, "required_free_vram_mb": 22000,
                  "idle_timeout_seconds": 0.02},
        "fallback": {"allow_runtime_restart": False},
        "lock": {"path": str(Path(temp_dir) / "manager.lock"), "timeout_seconds": 0.05},
        "state": {"path": str(Path(temp_dir) / "state.json")},
        "logging": {"level": "INFO"},
    }


class FakeEnvironment:
    def __init__(self):
        self.llama_state = "unloaded"
        self.llama_reachable = True
        self.comfy_reachable = True
        self.video_loaded = False
        self.comfy_running = 0
        self.comfy_pending = 0
        self.load_fails = False
        self.unload_fails = False
        self.unload_polls_remaining = 0
        self.free_fails = False
        self.retain_video_memory = False
        self.rogue_process = False
        self.calls = []

    def monitor(self, _index):
        processes = []
        if self.llama_state == "loaded":
            processes.append({"pid": 101, "name": "llama-server", "used_vram_mb": 14000})
        if self.video_loaded or self.comfy_running:
            processes.append({"pid": 202, "name": "python", "command_line": "python main.py",
                              "cwd": "/opt/ComfyUI", "used_vram_mb": 15000})
        if self.rogue_process:
            processes.append({"pid": 303, "name": "python", "command_line": "python trainer.py",
                              "used_vram_mb": 12000})
        used = 500 + sum(item["used_vram_mb"] for item in processes)
        return {"gpu_index": 0, "name": "Fake GPU", "total_vram_mb": 24564,
                "used_vram_mb": used, "free_vram_mb": 24564 - used,
                "utilization_percent": 0, "temperature_c": 35,
                "processes": processes, "backend": "fake"}

    def http(self, base, path, method="GET", payload=None, timeout=5):
        del timeout
        self.calls.append((base, path, method, copy.deepcopy(payload)))
        if base == "http://llama":
            if not self.llama_reachable:
                raise RuntimeManagerError("llama unreachable")
            if path == "/health": return {"status": "ok"}
            if path == "/models":
                if self.llama_state == "unloading" and self.unload_polls_remaining > 0:
                    self.unload_polls_remaining -= 1
                    if self.unload_polls_remaining == 0:
                        self.llama_state = "unloaded"
                return {"data": [{"id": "qwen", "status": {"value": self.llama_state}}]}
            if path == "/models/load":
                if self.load_fails: raise RuntimeManagerError("load failed")
                self.llama_state = "loaded"; return {"success": True}
            if path == "/models/unload":
                if self.unload_fails: raise RuntimeManagerError("unload failed")
                self.llama_state = "unloading" if self.unload_polls_remaining else "unloaded"
                return {"success": True}
        if base == "http://comfy":
            if not self.comfy_reachable:
                raise RuntimeManagerError("ComfyUI unreachable")
            if path == "/queue":
                return {"queue_running": [[1]] * self.comfy_running,
                        "queue_pending": [[2]] * self.comfy_pending}
            if path == "/system_stats": return {"system": {"comfyui_version": "test"}}
            if path.startswith("/history"): return {}
            if path == "/free":
                if self.free_fails: raise RuntimeManagerError("free failed")
                if not self.retain_video_memory: self.video_loaded = False
                return {}
        raise RuntimeManagerError(f"unexpected request {base}{path}")
