---
name: gpu-runtime-manager
description: Safely manage exclusive ownership of one NVIDIA GPU between a llama.cpp LLM and ComfyUI video generation. Use before Qwen reasoning, before MiniMax H3 generation, after video generation, when returning to the LLM, or whenever GPU/VRAM ownership, runtime health, queue state, unloading, loading, or switching must be checked.
---

# GPU Runtime Manager

## Purpose

Manage VRAM ownership in a single-GPU environment. Keep llama.cpp/Qwen and ComfyUI/MiniMax H3 mutually exclusive. Use the high-level manager instead of composing low-level HTTP, process, and GPU commands.

## When to use

Use this skill before every GPU-heavy workload, especially before using Qwen, before submitting a MiniMax H3 workflow, after video generation, and before returning to Qwen.

## Mandatory rules

- Before executing a GPU-heavy model, acquire the corresponding runtime.
- Never start MiniMax H3 while Qwen still owns the required VRAM.
- Never start Qwen while ComfyUI still owns the required VRAM.
- Never assume memory has been released only because an unload command succeeded.
- Always let the manager verify actual GPU memory usage through NVML or `nvidia-smi`.
- If GPU state is `UNKNOWN`, stop and report the status instead of loading another model.
- For Creative Agent video requests, acquire LLM first and ask the configured llama.cpp model to create the complete H3 prompt before acquiring VIDEO. Use that model output as the workflow prompt; do not silently replace it with an agent-authored prompt.
- Treat `acquire video` as a reservation and submit the intended ComfyUI workflow immediately afterward.
- Run commands from the repository root. Use `--config PATH` before the subcommand only when selecting another config.

## Agent commands

Inspect without changing anything:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py status
python skills/gpu-runtime-manager/scripts/runtime_manager.py status --json
```

Preview a transition:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py acquire video --dry-run
```

Use the LLM:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py acquire llm
```

Use the video model, then submit the workflow through the separate ComfyUI tool:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py acquire video
```

After video generation:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py release video
```

Return to the LLM:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py acquire llm
```

Explicit switches are also available:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py switch llm video
python skills/gpu-runtime-manager/scripts/runtime_manager.py switch video llm
```

## Required workflow

1. Run `status` if ownership is not already known.
2. For creative video generation, run `acquire llm`, ask the configured llama.cpp model to produce the complete H3 prompt, and retain its output for submission.
3. Run `acquire video` before the ComfyUI submission and use the retained llama.cpp output as the workflow prompt.
4. Stop if a manager command fails. Read the structured error and report the measured GPU processes and runtime states.
5. Execute only the acquired workload.
6. Run `release video` after ComfyUI finishes. Acquire LLM again before critique or planning.

The manager serializes transitions with `fcntl.flock`. It detects ownership from measured GPU processes, llama model state, and ComfyUI queue state. The state file is only a reservation hint; measured evidence wins.

## Failure handling

- For `UNKNOWN`, run `status --json` and report the contradiction or unexpected process. Do not retry with low-level load commands.
- For `workflow_still_running`, wait for the workflow to finish and retry. Never interrupt it merely to reclaim VRAM.
- For `vram_timeout`, report actual free VRAM and all GPU processes. Never cause an OOM to force eviction.
- For an unavailable runtime, report it. Runtime restart is allowed only when enabled in config and only for a llama-server process started and PID-recorded by this manager.
- Tune `required_free_vram_mb` in `config.yaml` after measuring the installed models. Values larger than total VRAM are rejected before mutation.

## Forbidden behavior

- Starting both heavy runtimes simultaneously.
- Blindly killing GPU processes.
- Assuming unload means VRAM is free.
- Ignoring `UNKNOWN` state.
- Triggering an OOM intentionally.
- Unloading ComfyUI while a workflow is running.
- Calling `/free`, `/models/load`, `/models/unload`, or process signals directly when the manager can perform the transition.

## Creative Agent example

```text
User: "15秒のアニメPVを作って"

acquire llm
  -> Qwen via llama.cpp generates the storyboard and complete H3 prompt
  -> acquire video
  -> manager unloads Qwen and verifies VRAM
  -> submit MiniMax H3 workflow through ComfyUI
  -> release video
  -> acquire llm
  -> video critique
  -> bad: acquire video, regenerate
  -> good: deliver final.mp4
```

## Low-level diagnostics

Use these only for diagnosis; do not replace the high-level acquire/release flow:

```bash
python skills/gpu-runtime-manager/scripts/gpu_status.py --json
python skills/gpu-runtime-manager/scripts/llama_status.py --json
python skills/gpu-runtime-manager/scripts/comfy_status.py --json
python skills/gpu-runtime-manager/scripts/wait_vram.py --free-mb 21000 --timeout 120
```
