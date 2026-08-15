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
- When OpenCode itself runs on the managed llama.cpp/Qwen model, use only the atomic `run-video` command for video generation. Never split VIDEO acquisition, submission, waiting, release, and LLM restoration across separate agent turns.
- Treat standalone `acquire video` as an external-controller interface. Do not use it from an OpenCode session whose next inference also depends on the managed Qwen.
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

Run video generation atomically when OpenCode uses the managed Qwen as its own model. Create the ComfyUI API-format workflow JSON while LLM is acquired, then make this the final tool call of the current Qwen turn:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py run-video \
  --workflow /tmp/minimax-h3-api-workflow.json
```

The command holds the GPU lock while it unloads Qwen, verifies VRAM, submits and waits for that exact workflow, frees ComfyUI memory, verifies VRAM again, reloads Qwen, and only then returns control to OpenCode. Use `--json` for structured output or override the configured completion limit with `--timeout SECONDS`.

Preview the complete atomic operation without changing runtime state or submitting the workflow:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py run-video \
  --workflow /tmp/minimax-h3-api-workflow.json \
  --dry-run
```

Use standalone ownership commands only from an external controller that does not depend on this Qwen for its next action:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py acquire video
python skills/gpu-runtime-manager/scripts/runtime_manager.py release video
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
3. Write a ComfyUI API-format workflow JSON containing that prompt while Qwen is still loaded. Do not pass a UI-format workflow export.
4. If OpenCode uses this Qwen as its own model, invoke `run-video --workflow FILE` once. Do not call any other agent tool until it returns; the command performs the entire VIDEO lease and restores LLM internally.
5. Stop if a manager command fails. Read the structured error and report the measured GPU processes and runtime states.
6. Resume critique or planning only after the returned state reports `gpu_owner=llm` and `llama.model_loaded=true`.

The manager serializes transitions with `fcntl.flock`. It detects ownership from measured GPU processes, llama model state, and ComfyUI queue state. The state file is only a reservation hint; measured evidence wins.

## Failure handling

- For `UNKNOWN`, run `status --json` and report the contradiction or unexpected process. Do not retry with low-level load commands.
- For `workflow_still_running`, wait for the workflow to finish and retry. Never interrupt it merely to reclaim VRAM.
- For `atomic_recovery_failure`, do not request another model or manually load Qwen. A workflow may still own the GPU. Inspect `status --json` from a human-controlled terminal and recover only after ComfyUI is idle.
- For `workflow_timeout`, use the returned measured state. The atomic command restores Qwen only when ComfyUI can be released safely.
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
- Calling standalone `acquire video` from OpenCode when OpenCode's next inference requires the same managed Qwen.
- Calling `/free`, `/models/load`, `/models/unload`, or process signals directly when the manager can perform the transition.

## Creative Agent example

```text
User: "15秒のアニメPVを作って"

acquire llm
  -> Qwen via llama.cpp generates the storyboard and complete H3 prompt
  -> write ComfyUI API workflow JSON with that prompt
  -> run-video --workflow ... (one OpenCode tool call)
     -> manager unloads Qwen and verifies VRAM
     -> manager submits and waits for MiniMax H3
     -> manager releases VIDEO and verifies VRAM
     -> manager reloads Qwen before returning
  -> video critique
  -> bad: create another workflow, then run-video once
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
