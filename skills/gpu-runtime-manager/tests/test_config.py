import tempfile
import unittest
from pathlib import Path

import yaml

from helpers import config
from gpu_runtime import ConfigurationError, GPUOwner, RuntimeManager, load_config


class ConfigTests(unittest.TestCase):
    def test_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"; path.write_text("gpu: {}\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError): load_config(path)

    def test_invalid_value(self):
        with tempfile.TemporaryDirectory() as directory:
            data = config(directory); data["gpu"]["poll_interval_seconds"] = 0
            path = Path(directory) / "bad.yaml"; path.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaises(ConfigurationError): load_config(path)

    def test_required_vram_cannot_exceed_gpu_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            data = config(directory); data["video"]["required_free_vram_mb"] = 99999
            monitor = lambda _: {"gpu_index": 0, "name": "GPU", "total_vram_mb": 24564,
                "used_vram_mb": 0, "free_vram_mb": 24564, "utilization_percent": 0,
                "temperature_c": 30, "processes": [], "backend": "fake"}
            def http(base, path, *args):
                if path == "/health": return {"status": "ok"}
                if path == "/models": return {"data": [{"id": "qwen", "status": {"value": "unloaded"}}]}
                if path == "/queue": return {"queue_running": [], "queue_pending": []}
                if path == "/system_stats": return {"system": {}}
                if path.startswith("/history"): return {}
                raise AssertionError(path)
            manager = RuntimeManager(data, monitor, http)
            with self.assertRaises(ConfigurationError): manager.acquire(GPUOwner.VIDEO)


if __name__ == "__main__":
    unittest.main()
