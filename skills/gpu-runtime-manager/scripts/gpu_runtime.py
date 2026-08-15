#!/usr/bin/env python3
"""Safe single-GPU ownership primitives for llama.cpp and ComfyUI."""
from __future__ import annotations

import contextlib
import enum
import fcntl
import json
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = SKILL_DIR / "config.yaml"
LOG = logging.getLogger("gpu-runtime-manager")


class GPUOwner(str, enum.Enum):
    NONE = "none"
    LLM = "llm"
    VIDEO = "video"
    UNKNOWN = "unknown"
    TRANSITIONING = "transitioning"


class RuntimeManagerError(RuntimeError):
    code = "runtime_error"


class ConfigurationError(RuntimeManagerError): code = "configuration_error"
class GPUMonitoringError(RuntimeManagerError): code = "gpu_monitoring_unavailable"
class VRAMTimeout(RuntimeManagerError): code = "vram_timeout"
class LlamaUnavailable(RuntimeManagerError): code = "llama_server_unavailable"
class LlamaOperationError(RuntimeManagerError): code = "llama_operation_failure"
class ComfyUnavailable(RuntimeManagerError): code = "comfyui_unavailable"
class ComfyOperationError(RuntimeManagerError): code = "comfyui_operation_failure"
class WorkflowBusy(RuntimeManagerError): code = "workflow_still_running"
class WorkflowSubmissionError(RuntimeManagerError): code = "workflow_submission_failure"
class WorkflowExecutionError(RuntimeManagerError): code = "workflow_execution_failure"
class WorkflowTimeout(RuntimeManagerError): code = "workflow_timeout"
class AtomicRecoveryError(RuntimeManagerError): code = "atomic_recovery_failure"
class UnexpectedGPUProcess(RuntimeManagerError): code = "unexpected_gpu_process"
class UnknownGPUState(RuntimeManagerError): code = "unknown_gpu_state"
class LockTimeout(RuntimeManagerError): code = "lock_timeout"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if yaml is None:
        raise ConfigurationError("PyYAML is required")
    cfg_path = Path(path or os.environ.get("GPU_RUNTIME_MANAGER_CONFIG", DEFAULT_CONFIG))
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read config {cfg_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("config root must be a mapping")
    required = [
        ("gpu", "device"), ("gpu", "poll_interval_seconds"),
        ("gpu", "transition_timeout_seconds"), ("llm", "base_url"),
        ("llm", "model"), ("llm", "required_free_vram_mb"),
        ("video", "base_url"), ("video", "model"),
        ("video", "required_free_vram_mb"), ("lock", "path"),
    ]
    for section, key in required:
        if not isinstance(data.get(section), dict) or key not in data[section]:
            raise ConfigurationError(f"missing config: {section}.{key}")
    for runtime in ("llm", "video"):
        if not isinstance(data[runtime]["model"], dict) or not data[runtime]["model"].get("name"):
            raise ConfigurationError(f"{runtime}.model.name must be a non-empty string")
    if not isinstance(data["gpu"]["device"], int) or data["gpu"]["device"] < 0:
        raise ConfigurationError("gpu.device must be a non-negative integer")
    for section, key in [
        ("gpu", "poll_interval_seconds"), ("gpu", "transition_timeout_seconds"),
        ("llm", "required_free_vram_mb"), ("video", "required_free_vram_mb"),
    ]:
        value = data[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigurationError(f"{section}.{key} must be positive")
    workflow_timeout = data["video"].get("workflow_timeout_seconds", 3600)
    if (isinstance(workflow_timeout, bool)
            or not isinstance(workflow_timeout, (int, float))
            or workflow_timeout <= 0):
        raise ConfigurationError("video.workflow_timeout_seconds must be positive")
    for section, key in (("lock", "path"), ("state", "path")):
        value = data.get(section, {}).get(key)
        if value is not None and (not isinstance(value, str) or not Path(value).is_absolute()):
            raise ConfigurationError(f"{section}.{key} must be an absolute path")
    if data["llm"].get("management_mode", "auto") not in {"auto", "router", "managed-process"}:
        raise ConfigurationError("llm.management_mode must be auto, router, or managed-process")
    return data


def setup_logging(cfg: dict[str, Any], json_mode: bool = False) -> None:
    name = str(cfg.get("logging", {}).get("level", "INFO")).upper()
    level = getattr(logging, name, None)
    if not isinstance(level, int):
        raise ConfigurationError(f"invalid logging.level: {name}")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)
    if json_mode:
        LOG.setLevel(max(level, logging.WARNING))


def _process_details(pid: int, fallback_name: str) -> dict[str, Any]:
    details: dict[str, Any] = {"pid": pid, "name": fallback_name}
    with contextlib.suppress(OSError):
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").strip()
        command_line = raw.decode(errors="replace")
        details["command_line"] = command_line[:512] + ("..." if len(command_line) > 512 else "")
    with contextlib.suppress(OSError):
        details["cwd"] = os.readlink(f"/proc/{pid}/cwd")
    with contextlib.suppress(OSError):
        details["executable"] = os.readlink(f"/proc/{pid}/exe")
    return details


def _nvml_status(index: int) -> dict[str, Any]:
    try:
        import pynvml  # type: ignore
    except ImportError as exc:
        raise GPUMonitoringError("pynvml is not installed") from exc
    initialized = False
    try:
        pynvml.nvmlInit(); initialized = True
        if pynvml.nvmlDeviceGetCount() != 1:
            raise GPUMonitoringError("this skill requires exactly one visible GPU")
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        processes, seen = [], set()
        for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                       pynvml.nvmlDeviceGetGraphicsRunningProcesses):
            try: rows = getter(handle)
            except pynvml.NVMLError: rows = []
            for process in rows:
                if process.pid in seen: continue
                seen.add(process.pid)
                try: name = pynvml.nvmlSystemGetProcessName(process.pid)
                except pynvml.NVMLError: name = "unknown"
                if isinstance(name, bytes): name = name.decode(errors="replace")
                item = _process_details(process.pid, str(name))
                item["used_vram_mb"] = round(getattr(process, "usedGpuMemory", 0) / 1048576)
                processes.append(item)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes): name = name.decode(errors="replace")
        return {"gpu_index": index, "name": name,
                "total_vram_mb": round(memory.total / 1048576),
                "used_vram_mb": round(memory.used / 1048576),
                "free_vram_mb": round(memory.free / 1048576),
                "utilization_percent": utilization.gpu,
                "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, 0),
                "processes": processes, "backend": "nvml"}
    except GPUMonitoringError:
        raise
    except Exception as exc:
        raise GPUMonitoringError(f"NVML failed: {exc}") from exc
    finally:
        if initialized:
            with contextlib.suppress(Exception): pynvml.nvmlShutdown()


def _smi_status(index: int) -> dict[str, Any]:
    fields = "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
    try:
        count = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10, check=True)
        rows = [line for line in count.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError(f"this skill requires exactly one visible GPU; found {len(rows)}")
        result = subprocess.run(["nvidia-smi", f"--id={index}", f"--query-gpu={fields}",
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=10, check=True)
        rows = result.stdout.strip().splitlines()
        if len(rows) != 1: raise ValueError(f"expected one GPU row, got {len(rows)}")
        values = [value.strip() for value in rows[0].split(",")]
        if len(values) != 7: raise ValueError("unexpected nvidia-smi GPU output")
        proc_result = subprocess.run(
            ["nvidia-smi", f"--id={index}",
             "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10, check=True)
        processes = []
        for line in proc_result.stdout.splitlines():
            parts = [part.strip() for part in line.rsplit(",", 2)]
            if len(parts) != 3: continue
            try: pid, used = int(parts[0]), int(parts[2])
            except ValueError: continue
            item = _process_details(pid, parts[1]); item["used_vram_mb"] = used
            processes.append(item)
        return {"gpu_index": int(values[0]), "name": values[1],
                "total_vram_mb": int(values[2]), "used_vram_mb": int(values[3]),
                "free_vram_mb": int(values[4]), "utilization_percent": int(values[5]),
                "temperature_c": int(values[6]), "processes": processes,
                "backend": "nvidia-smi"}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise GPUMonitoringError(f"nvidia-smi failed: {exc}") from exc


def gpu_status(index: int) -> dict[str, Any]:
    try: return _nvml_status(index)
    except GPUMonitoringError as nvml_error:
        try: return _smi_status(index)
        except GPUMonitoringError as smi_error:
            raise GPUMonitoringError(f"GPU monitoring failed ({nvml_error}; {smi_error})") from smi_error


def wait_for_vram(index: int, free_mb: int, timeout: float, interval: float,
                  monitor: Callable[[int], dict[str, Any]] = gpu_status) -> dict[str, Any]:
    if index < 0 or free_mb <= 0 or timeout <= 0 or interval <= 0:
        raise ConfigurationError("gpu, free-mb, timeout, and poll interval must be positive")
    deadline = time.monotonic() + timeout
    while True:
        last = monitor(index)
        if last["free_vram_mb"] >= free_mb: return last
        if time.monotonic() >= deadline:
            raise VRAMTimeout(f"timed out waiting for {free_mb} MB free; actual={last['free_vram_mb']} MB")
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))


def request_json(base: str, path: str, method: str = "GET",
                 payload: dict[str, Any] | None = None, timeout: float = 5) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeManagerError(f"HTTP {method} {path} failed: {exc}") from exc


def llama_status(cfg: dict[str, Any], http: Callable[..., Any] = request_json) -> dict[str, Any]:
    base, wanted = cfg["llm"]["base_url"], cfg["llm"]["model"]["name"]
    result: dict[str, Any] = {"reachable": False, "healthy": False, "router_mode": False,
        "model_available": False, "model_loaded": False, "model_identifier": wanted,
        "model_state": "unavailable", "sleeping": None, "models": [],
        "foreign_active_models": []}
    try:
        health = http(base, "/health")
        result.update(reachable=True, healthy=health.get("status") == "ok")
    except RuntimeManagerError:
        return result
    try:
        models = http(base, "/models")
        rows = models.get("data", []) if isinstance(models, dict) else []
        router_rows = [row for row in rows if isinstance(row, dict) and "status" in row]
        if router_rows:
            match = next((row for row in router_rows if row.get("id") == wanted), None)
            foreign = [row.get("id") for row in router_rows if row.get("id") != wanted
                       and row.get("status", {}).get("value") in {"loaded", "loading"}]
            result.update(router_mode=True, models=router_rows, foreign_active_models=foreign)
            if match:
                state = match.get("status", {}).get("value", "unknown")
                result.update(model_available=True, model_loaded=state in {"loaded", "sleeping"},
                              model_state=state, sleeping=state == "sleeping")
            else: result["model_state"] = "not_found"
            return result
    except RuntimeManagerError:
        pass
    try:
        props = http(base, "/props")
        sleeping = props.get("is_sleeping")
        result.update(model_available=True, model_loaded=True,
                      model_identifier=props.get("model_alias") or props.get("model_path"),
                      model_state="sleeping" if sleeping else "loaded", sleeping=sleeping,
                      props=props)
    except RuntimeManagerError:
        result["healthy"] = False
    return result


def comfy_status(cfg: dict[str, Any], http: Callable[..., Any] = request_json) -> dict[str, Any]:
    base = cfg["video"]["base_url"]
    result: dict[str, Any] = {"reachable": False, "healthy": False,
        "running_workflow_count": 0, "pending_workflow_count": 0,
        "queue_length": 0, "idle": False, "version": None}
    try:
        queue = http(base, "/queue")
        running, pending = queue.get("queue_running", []), queue.get("queue_pending", [])
        result.update(reachable=True, healthy=True, running_workflow_count=len(running),
                      pending_workflow_count=len(pending), queue_length=len(running) + len(pending),
                      idle=not running and not pending, queue=queue)
        with contextlib.suppress(RuntimeManagerError):
            stats = http(base, "/system_stats")
            result.update(system_stats=stats, version=stats.get("system", {}).get("comfyui_version"))
        with contextlib.suppress(RuntimeManagerError):
            history = http(base, "/history?max_items=1")
            result["latest_history"] = [
                {"prompt_id": prompt_id,
                 "status": item.get("status", {}).get("status_str"),
                 "completed": item.get("status", {}).get("completed", False)}
                for prompt_id, item in history.items()
                if isinstance(item, dict)
            ] if isinstance(history, dict) else []
    except RuntimeManagerError:
        pass
    return result


def classify_process(process: dict[str, Any], cfg: dict[str, Any]) -> str:
    text = " ".join(str(process.get(key, "")) for key in
                    ("name", "command_line", "cwd", "executable")).lower()
    detection = cfg.get("process_detection", {})
    llm = any(str(x).lower() in text for x in detection.get("llm_patterns", ["llama-server", "llama.cpp"]))
    video = any(str(x).lower() in text for x in detection.get("video_patterns", ["comfyui"]))
    if llm and video: return "ambiguous"
    if llm: return "llm"
    if video: return "video"
    return "other"


def detect_owner(gpu: dict[str, Any], llama: dict[str, Any], comfy: dict[str, Any],
                 cfg: dict[str, Any]) -> tuple[GPUOwner, list[dict[str, Any]]]:
    threshold = int(cfg["gpu"].get("owner_process_threshold_mb", 1024))
    heavy = [p for p in gpu.get("processes", []) if p.get("used_vram_mb", 0) >= threshold]
    llm_procs = [p for p in heavy if classify_process(p, cfg) == "llm"]
    video_procs = [p for p in heavy if classify_process(p, cfg) == "video"]
    unexpected = [p for p in heavy if classify_process(p, cfg) not in {"llm", "video"}]
    llm_active = bool(llm_procs) or bool(llama.get("model_loaded") and llama.get("sleeping") is not True)
    video_active = bool(video_procs) or bool(comfy.get("running_workflow_count"))
    transitional = llama.get("model_state") in {"loading", "unloading"}
    if unexpected or llama.get("foreign_active_models") or transitional or (llm_active and video_active):
        return GPUOwner.UNKNOWN, unexpected
    if llm_active: return GPUOwner.LLM, unexpected
    if video_active: return GPUOwner.VIDEO, unexpected
    return GPUOwner.NONE, unexpected


class FileLock:
    def __init__(self, path: str, timeout: float):
        self.path, self.timeout, self.file = path, timeout, None

    def __enter__(self) -> "FileLock":
        try: self.file = open(self.path, "a+", encoding="utf-8")
        except OSError as exc: raise LockTimeout(f"cannot open lock {self.path}: {exc}") from exc
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB); return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.file.close(); raise LockTimeout(f"timed out acquiring lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *_: Any) -> None:
        if self.file:
            fcntl.flock(self.file, fcntl.LOCK_UN); self.file.close()


@dataclass
class RuntimeManager:
    cfg: dict[str, Any]
    monitor: Callable[[int], dict[str, Any]] = gpu_status
    http: Callable[..., Any] = request_json

    def _read_persisted_owner(self) -> GPUOwner | None:
        path = Path(self.cfg.get("state", {}).get("path", "/tmp/gpu-runtime-manager-state.json"))
        try: return GPUOwner(json.loads(path.read_text(encoding="utf-8"))["owner"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError): return None

    def snapshot(self) -> dict[str, Any]:
        gpu = self.monitor(self.cfg["gpu"]["device"])
        llama, comfy = llama_status(self.cfg, self.http), comfy_status(self.cfg, self.http)
        detected, unexpected = detect_owner(gpu, llama, comfy, self.cfg)
        persisted = self._read_persisted_owner()
        owner = persisted if detected is GPUOwner.NONE and persisted in {GPUOwner.LLM, GPUOwner.VIDEO} else detected
        return {"gpu_owner": owner.value, "detected_state": detected.value,
                "persisted_owner": persisted.value if persisted else None,
                "state_mismatch": bool(persisted in {GPUOwner.LLM, GPUOwner.VIDEO}
                                       and detected in {GPUOwner.LLM, GPUOwner.VIDEO}
                                       and persisted is not detected),
                "gpu": gpu, "llama": llama, "comfyui": comfy,
                "unexpected_processes": unexpected}

    def _persist(self, owner: GPUOwner) -> None:
        path = Path(self.cfg.get("state", {}).get("path", "/tmp/gpu-runtime-manager-state.json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps({"owner": owner.value, "updated_at": time.time(), "pid": os.getpid()}),
                        encoding="utf-8")
        os.replace(temp, path)

    def _validate_capacity(self, snapshot: dict[str, Any], target: GPUOwner) -> None:
        required, total = int(self.cfg[target.value]["required_free_vram_mb"]), int(snapshot["gpu"]["total_vram_mb"])
        if required > total:
            raise ConfigurationError(f"{target.value}.required_free_vram_mb={required} exceeds total VRAM {total} MB")

    def _wait_vram(self, amount: int) -> dict[str, Any]:
        LOG.info("waiting_for_vram free_required=%sMB", amount)
        result = wait_for_vram(self.cfg["gpu"]["device"], amount,
            self.cfg["gpu"]["transition_timeout_seconds"],
            self.cfg["gpu"]["poll_interval_seconds"], self.monitor)
        LOG.info("gpu_free=%sMB", result["free_vram_mb"]); return result

    def _wait_comfy_idle(self) -> dict[str, Any]:
        timeout = self.cfg["video"].get("idle_timeout_seconds", self.cfg["gpu"]["transition_timeout_seconds"])
        deadline = time.monotonic() + timeout
        while True:
            status = comfy_status(self.cfg, self.http)
            if not status["reachable"]: raise ComfyUnavailable("ComfyUI is unavailable")
            if status["idle"]: return status
            if time.monotonic() >= deadline: raise WorkflowBusy("ComfyUI did not become idle before timeout")
            time.sleep(self.cfg["gpu"]["poll_interval_seconds"])

    def _free_comfy(self) -> None:
        LOG.info("waiting_for_comfyui_idle"); self._wait_comfy_idle()
        LOG.info("requesting_comfyui_free")
        try:
            self.http(self.cfg["video"]["base_url"], "/free", "POST",
                      {"unload_models": True, "free_memory": True})
        except RuntimeManagerError as exc: raise ComfyOperationError(str(exc)) from exc
        after = comfy_status(self.cfg, self.http)
        if not after["reachable"]: raise ComfyUnavailable("ComfyUI became unavailable after cleanup")
        if not after["idle"]: raise WorkflowBusy("ComfyUI became busy during memory cleanup")

    def _submit_comfy_workflow(self, workflow: dict[str, Any]) -> str:
        if not isinstance(workflow, dict) or not workflow:
            raise ConfigurationError("workflow must be a non-empty JSON object")
        body = workflow if "prompt" in workflow else {"prompt": workflow}
        if not isinstance(body.get("prompt"), dict) or not body["prompt"]:
            raise ConfigurationError("workflow.prompt must be a non-empty ComfyUI API graph")
        LOG.info("submitting_comfyui_workflow")
        try:
            response = self.http(self.cfg["video"]["base_url"], "/prompt", "POST", body)
        except RuntimeManagerError as exc:
            raise WorkflowSubmissionError(str(exc)) from exc
        if not isinstance(response, dict) or not response.get("prompt_id"):
            raise WorkflowSubmissionError(f"ComfyUI did not return prompt_id: {response}")
        if response.get("node_errors"):
            raise WorkflowSubmissionError(f"ComfyUI rejected workflow: {response['node_errors']}")
        prompt_id = str(response["prompt_id"])
        LOG.info("comfyui_workflow_submitted prompt_id=%s", prompt_id)
        return prompt_id

    @staticmethod
    def _workflow_files(outputs: Any) -> list[dict[str, Any]]:
        files: list[dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("filename"), str):
                    files.append({key: value[key] for key in ("filename", "subfolder", "type")
                                  if key in value})
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(outputs)
        return files

    def _wait_comfy_workflow(self, prompt_id: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        path = f"/history/{prompt_id}"
        while True:
            try:
                history = self.http(self.cfg["video"]["base_url"], path)
            except RuntimeManagerError as exc:
                raise ComfyUnavailable(f"ComfyUI became unavailable while waiting: {exc}") from exc
            item = history.get(prompt_id) if isinstance(history, dict) else None
            if isinstance(item, dict):
                status = item.get("status", {})
                if status.get("completed"):
                    if status.get("status_str") != "success":
                        raise WorkflowExecutionError(
                            f"ComfyUI workflow {prompt_id} failed: {status}"
                        )
                    LOG.info("comfyui_workflow_completed prompt_id=%s", prompt_id)
                    return {"prompt_id": prompt_id, "status": "success",
                            "files": self._workflow_files(item.get("outputs", {}))}
            if time.monotonic() >= deadline:
                raise WorkflowTimeout(
                    f"timed out waiting for ComfyUI workflow {prompt_id} after {timeout:g} seconds"
                )
            time.sleep(min(self.cfg["gpu"]["poll_interval_seconds"],
                           max(0.0, deadline - time.monotonic())))

    def _llama_unload(self) -> None:
        status = llama_status(self.cfg, self.http)
        if not status["reachable"]: raise LlamaUnavailable("llama-server is unavailable")
        if not status["model_loaded"] and status.get("model_state") != "loading": return
        if status["router_mode"]:
            LOG.info("requesting_llama_unload model=%s", self.cfg["llm"]["model"]["name"])
            try:
                response = self.http(self.cfg["llm"]["base_url"], "/models/unload", "POST",
                                     {"model": self.cfg["llm"]["model"]["name"]})
            except RuntimeManagerError as exc: raise LlamaOperationError(str(exc)) from exc
            if isinstance(response, dict) and response.get("success") is False:
                raise LlamaOperationError(f"llama unload rejected: {response}")
            deadline = time.monotonic() + self.cfg["gpu"]["transition_timeout_seconds"]
            last = status
            while time.monotonic() < deadline:
                last = llama_status(self.cfg, self.http)
                if (last["reachable"] and not last["model_loaded"]
                        and last.get("model_state") not in {"loading", "unloading"}):
                    return
                time.sleep(self.cfg["gpu"]["poll_interval_seconds"])
            raise LlamaOperationError(
                f"llama model did not finish unloading before timeout; last_status={last}"
            )
        self._stop_managed_llama()

    def _llama_load(self) -> None:
        status, mode = llama_status(self.cfg, self.http), self.cfg["llm"].get("management_mode", "auto")
        if status["router_mode"]:
            if not status["model_available"]:
                raise LlamaUnavailable(f"configured model {self.cfg['llm']['model']['name']!r} is not available in /models")
            # This installed llama.cpp has idle sleep, but no explicit wake endpoint.
            if status.get("sleeping"): self._llama_unload()
            elif status["model_loaded"] and status["model_state"] == "loaded": return
            LOG.info("requesting_llama_load model=%s", self.cfg["llm"]["model"]["name"])
            try:
                response = self.http(self.cfg["llm"]["base_url"], "/models/load", "POST",
                                     {"model": self.cfg["llm"]["model"]["name"]})
            except RuntimeManagerError as exc: raise LlamaOperationError(str(exc)) from exc
            if isinstance(response, dict) and response.get("success") is False:
                raise LlamaOperationError(f"llama load rejected: {response}")
        elif mode == "router": raise LlamaUnavailable("llama.cpp router API is unavailable")
        elif status["reachable"] and status["healthy"] and status["model_loaded"]:
            if status.get("sleeping"):
                raise LlamaOperationError("single-model server is sleeping and has no explicit wake API")
            return
        else: self._start_managed_llama()
        deadline, last = time.monotonic() + self.cfg["llm"].get("startup_timeout_seconds", 120), None
        while time.monotonic() < deadline:
            last = llama_status(self.cfg, self.http)
            if last["healthy"] and last["model_loaded"] and not last.get("sleeping"): return
            time.sleep(self.cfg["gpu"]["poll_interval_seconds"])
        raise LlamaOperationError(f"llama model did not become ready before timeout; last_status={last}")

    def _managed(self) -> dict[str, Any]: return self.cfg["llm"].get("managed_process", {})

    def _stop_managed_llama(self) -> None:
        if not self.cfg.get("fallback", {}).get("allow_runtime_restart", False):
            raise LlamaOperationError("managed restart fallback is disabled")
        pid_file = Path(self._managed().get("pid_file", ""))
        if not pid_file.is_file():
            raise LlamaOperationError("refusing to stop llama-server without a manager-owned PID file")
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except (OSError, ValueError) as exc: raise LlamaOperationError("invalid managed llama PID file") from exc
        if "llama-server" not in cmdline: raise LlamaOperationError("managed PID is not llama-server")
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
        except OSError as exc: raise LlamaOperationError(f"cannot stop managed llama-server: {exc}") from exc
        deadline = time.monotonic() + 10
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline: time.sleep(0.1)
        if Path(f"/proc/{pid}").exists():
            raise LlamaOperationError("managed llama-server did not stop; refusing SIGKILL")
        pid_file.unlink(missing_ok=True)

    def _start_managed_llama(self) -> None:
        if not self.cfg.get("fallback", {}).get("allow_runtime_restart", False):
            raise LlamaUnavailable("runtime restart fallback is disabled")
        managed, command = self._managed(), self._managed().get("command", [])
        if not isinstance(command, list) or not command:
            raise LlamaUnavailable("managed llama startup command is not configured")
        try:
            log = open(Path(managed.get("log_file", "/tmp/gpu-runtime-manager-llama.log")), "ab")
            process = subprocess.Popen(command, cwd=managed.get("cwd") or None, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            log.close(); Path(managed["pid_file"]).write_text(str(process.pid), encoding="utf-8")
        except OSError as exc: raise LlamaUnavailable(f"cannot start managed llama-server: {exc}") from exc

    def acquire(self, target: GPUOwner, dry_run: bool = False) -> dict[str, Any]:
        if target not in {GPUOwner.LLM, GPUOwner.VIDEO}: raise ConfigurationError("target must be llm or video")
        before = self.snapshot(); self._validate_capacity(before, target)
        if before["unexpected_processes"]:
            raise UnexpectedGPUProcess("unexpected heavy GPU process detected; refusing transition")
        if before["detected_state"] == GPUOwner.UNKNOWN.value:
            raise UnknownGPUState("GPU state is UNKNOWN; refusing to load another model")
        current, actions = GPUOwner(before["gpu_owner"]), []
        if target is GPUOwner.LLM:
            if current is GPUOwner.VIDEO:
                actions += ["wait for ComfyUI idle", "unload ComfyUI models and free memory", "verify actual VRAM release"]
            actions += [f"wait for {self.cfg['llm']['required_free_vram_mb']} MB free VRAM",
                        "load or wake Qwen", "verify llama-server readiness", "set owner LLM"]
        else:
            if current is GPUOwner.LLM: actions += ["unload Qwen", "verify actual VRAM release"]
            actions += [f"wait for {self.cfg['video']['required_free_vram_mb']} MB free VRAM",
                        "verify ComfyUI health", "set owner VIDEO"]
        if dry_run: return {"dry_run": True, "target": target.value, "current": before, "would": actions}
        if target is GPUOwner.LLM:
            if current is GPUOwner.LLM and before["llama"]["healthy"] and before["llama"]["model_loaded"] and not before["llama"].get("sleeping"):
                return before
            self._persist(GPUOwner.TRANSITIONING)
            if current is GPUOwner.VIDEO: self._free_comfy()
            self._wait_vram(int(self.cfg["llm"]["required_free_vram_mb"])); self._llama_load()
        else:
            if current is GPUOwner.VIDEO and before["comfyui"]["healthy"]: return before
            self._persist(GPUOwner.TRANSITIONING)
            if current is GPUOwner.LLM: self._llama_unload()
            self._wait_vram(int(self.cfg["video"]["required_free_vram_mb"]))
            if not comfy_status(self.cfg, self.http)["reachable"]: raise ComfyUnavailable("ComfyUI is unavailable")
        if target is GPUOwner.VIDEO:
            verification = self.snapshot()
            if verification["unexpected_processes"]:
                raise UnexpectedGPUProcess("unexpected heavy GPU process appeared during transition")
            if verification["detected_state"] == GPUOwner.UNKNOWN.value:
                raise UnknownGPUState("GPU became UNKNOWN during VIDEO transition")
            if verification["llama"]["model_loaded"] or verification["detected_state"] == GPUOwner.LLM.value:
                raise LlamaOperationError("llama is still active after VIDEO transition")
        self._persist(target); result = self.snapshot()
        if target is GPUOwner.LLM and not (result["llama"]["healthy"] and result["llama"]["model_loaded"] and not result["llama"].get("sleeping")):
            raise LlamaOperationError("llama readiness verification failed")
        LOG.info("owner=%s", target.value.upper()); return result

    def release(self, target: GPUOwner, dry_run: bool = False) -> dict[str, Any]:
        if target not in {GPUOwner.LLM, GPUOwner.VIDEO}: raise ConfigurationError("target must be llm or video")
        before = self.snapshot()
        if before["unexpected_processes"]:
            raise UnexpectedGPUProcess("unexpected heavy GPU process detected; refusing transition")
        if before["detected_state"] == GPUOwner.UNKNOWN.value:
            raise UnknownGPUState("GPU state is UNKNOWN; refusing automatic release")
        current = GPUOwner(before["gpu_owner"])
        opposite = "video" if target is GPUOwner.LLM else "llm"
        required = int(self.cfg[opposite]["required_free_vram_mb"])
        total = int(before["gpu"]["total_vram_mb"])
        if required > total:
            raise ConfigurationError(
                f"{opposite}.required_free_vram_mb={required} exceeds total VRAM {total} MB"
            )
        actions = (["unload Qwen", "verify actual VRAM release", "set owner NONE"] if target is GPUOwner.LLM
                   else ["wait for ComfyUI idle", "unload ComfyUI models and free memory",
                         "verify actual VRAM release", "set owner NONE"])
        if dry_run: return {"dry_run": True, "target": target.value, "current": before, "would": actions}
        if current is not target: return before
        self._persist(GPUOwner.TRANSITIONING)
        if target is GPUOwner.LLM: self._llama_unload()
        else: self._free_comfy()
        self._wait_vram(required); self._persist(GPUOwner.NONE); result = self.snapshot()
        if result["detected_state"] != GPUOwner.NONE.value:
            raise RuntimeManagerError(f"runtime released but detected_state={result['detected_state']}")
        LOG.info("owner=NONE"); return result

    def run_video_workflow(self, workflow: dict[str, Any], dry_run: bool = False,
                           timeout: float | None = None) -> dict[str, Any]:
        before = self.snapshot()
        llama = before["llama"]
        if (before["detected_state"] != GPUOwner.LLM.value
                or not llama["healthy"]
                or not llama["model_loaded"]
                or llama.get("sleeping")):
            raise UnknownGPUState(
                "atomic run-video must start with healthy, measured LLM ownership "
                "so OpenCode can resume safely"
            )
        workflow_timeout = (self.cfg["video"].get("workflow_timeout_seconds", 3600)
                            if timeout is None else timeout)
        if (isinstance(workflow_timeout, bool)
                or not isinstance(workflow_timeout, (int, float))
                or workflow_timeout <= 0):
            raise ConfigurationError("workflow timeout must be positive")
        if not isinstance(workflow, dict) or not workflow:
            raise ConfigurationError("workflow must be a non-empty JSON object")
        body = workflow if "prompt" in workflow else {"prompt": workflow}
        if not isinstance(body.get("prompt"), dict) or not body["prompt"]:
            raise ConfigurationError("workflow.prompt must be a non-empty ComfyUI API graph")
        if dry_run:
            preview = self.acquire(GPUOwner.VIDEO, True)
            return {"dry_run": True, "target": "video_then_llm", "current": before,
                    "would": preview["would"] + [
                        "submit the ComfyUI API workflow",
                        f"wait up to {workflow_timeout:g} seconds for that exact workflow",
                        "wait for ComfyUI idle and free video models",
                        "verify actual VRAM release",
                        "load Qwen and verify llama-server readiness",
                        "return control to OpenCode only after owner is LLM",
                    ]}

        try:
            self.acquire(GPUOwner.VIDEO)
            prompt_id = self._submit_comfy_workflow(body)
            workflow_result = self._wait_comfy_workflow(prompt_id, float(workflow_timeout))
        except BaseException as primary_error:
            try:
                self.release(GPUOwner.VIDEO)
                self.acquire(GPUOwner.LLM)
            except Exception as recovery_error:
                raise AtomicRecoveryError(
                    f"video operation failed ({primary_error}); LLM recovery also failed "
                    f"({recovery_error}); do not request another model until status is inspected"
                ) from recovery_error
            raise

        try:
            self.release(GPUOwner.VIDEO)
            restored = self.acquire(GPUOwner.LLM)
        except Exception as recovery_error:
            raise AtomicRecoveryError(
                f"video workflow succeeded but LLM recovery failed: {recovery_error}"
            ) from recovery_error
        return {**restored, "atomic_video": workflow_result}

    def locked(self) -> FileLock:
        return FileLock(self.cfg["lock"]["path"], self.cfg["lock"].get("timeout_seconds", 30))


def error_payload(exc: Exception, manager: RuntimeManager | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": getattr(exc, "code", "runtime_error"),
                               "message": str(exc)}
    if manager:
        with contextlib.suppress(Exception): payload["actual"] = manager.snapshot()
    return payload
