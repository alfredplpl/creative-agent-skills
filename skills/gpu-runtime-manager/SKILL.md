---
name: gpu-runtime-manager
description: Safely manage exclusive ownership of one NVIDIA GPU between a llama.cpp LLM and ComfyUI video generation, including building and atomically running bundled MiniMax H3 text-to-video and single-image reference-to-video workflows. Use before Qwen reasoning, for MiniMax H3 generation from a prompt or character reference image, after video generation, when returning to the LLM, or whenever GPU/VRAM ownership, runtime health, queue state, unloading, loading, or switching must be checked.
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
- Save the Qwen-generated H3 prompt to a UTF-8 text file and pass it to `run-video --prompt-file`. Let the bundled builder create the ComfyUI graph; do not ask the agent to invent node IDs or model bindings.
- For character-reference generation, ask Qwen for an R2VA prompt containing the exact tag `<Picture 1>`, then pass exactly one reference through `--opencode-attachment` or `--reference-image`. Never substitute first-frame I2V when identity reference is requested.
- When the user attached the reference image to OpenCode, pass its displayed filename with `--opencode-attachment`. Do not use glob, `find`, Downloads, or the attachment's original source path. The manager resolves the active OpenCode session and stages the embedded attachment bytes itself.
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

Run video generation atomically when OpenCode uses the managed Qwen as its own model. Save Qwen's complete H3 prompt to a text file while LLM is acquired, then make this the final tool call of the current Qwen turn:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py run-video \
  --prompt-file /tmp/minimax-h3-prompt.txt \
  --width 864 \
  --height 480 \
  --duration 3 \
  --output-prefix video/creative_agent \
  --json
```

The command builds the bundled MiniMax H3 text-to-video API graph, holds the GPU lock while it unloads Qwen, verifies VRAM, submits and waits for that exact workflow, frees ComfyUI memory, verifies VRAM again, reloads Qwen, and only then returns control to OpenCode. Defaults are 864x480 and 3 seconds. Width and height must be multiples of 32; duration must be at most 15 seconds. Override the configured completion limit with `--timeout SECONDS`.

For a character fixed from one input image, have Qwen mention `<Picture 1>` in the prompt and use R2V:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py run-video \
  --prompt-file /tmp/minimax-h3-r2v-prompt.txt \
  --reference-image /absolute/path/to/character.png \
  --reference-quality max \
  --width 864 --height 480 --duration 5 \
  --output-prefix video/character_run \
  --json
```

The manager validates and uploads the image before unloading Qwen, then builds the installed `MiniMaxH3ReferenceToVideo` graph with the dedicated ref2va weights. Use `match` for lower reference cost or `max` for stronger identity fidelity and slower generation. Dry-run validates the local image path and graph but does not upload or submit anything.

When the image was attached in the current OpenCode conversation, use the attachment filename instead of looking for a filesystem copy:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py run-video \
  --prompt-file /tmp/minimax-h3-r2v-prompt.txt \
  --opencode-attachment character.png \
  --reference-quality max \
  --width 864 --height 480 --duration 5 \
  --output-prefix video/character_run \
  --json
```

Omit the filename after `--opencode-attachment` to select the newest user image in the active OpenCode session. The manager reads only the latest active session for the current repository, decodes the attachment's embedded data, uploads that exact content to ComfyUI, and removes its temporary staged copy. If multiple sessions are simultaneously active, stop on `opencode_attachment_failure`; do not search the filesystem for a same-named file.

Preview the complete atomic operation without changing runtime state or submitting the workflow:

```bash
python skills/gpu-runtime-manager/scripts/runtime_manager.py run-video \
  --prompt-file /tmp/minimax-h3-prompt.txt \
  --dry-run
```

Use `--workflow FILE` only for an intentionally custom ComfyUI API-format graph. The normal H3 path does not require OpenCode to understand ComfyUI nodes.

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
3. Save the exact prompt as a UTF-8 text file while Qwen is still loaded. For R2V, require `<Picture 1>` and select one reference image.
4. If OpenCode uses this Qwen as its own model, invoke `run-video --prompt-file FILE` once. Add `--opencode-attachment FILENAME` for a chat attachment or `--reference-image IMAGE` for an explicitly supplied local path. Do not call any other agent tool until it returns; the command builds the workflow, performs the entire VIDEO lease, and restores LLM internally.
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
- Passing a reference image to a prompt that omits `<Picture 1>`.
- Searching the filesystem for an OpenCode attachment or silently substituting a same-named local file.

## Creative Agent example

```text
User: "15秒のアニメPVを作って"

acquire llm
  -> Qwen via llama.cpp generates the storyboard and complete H3 prompt
  -> save the H3 prompt to a UTF-8 text file
  -> run-video --prompt-file ... (one OpenCode tool call)
     -> manager builds the bundled MiniMax H3 API workflow
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

To inspect or hand off the generated API workflow without running it:

```bash
python skills/gpu-runtime-manager/scripts/prepare_h3_workflow.py \
  --prompt-file /tmp/minimax-h3-prompt.txt \
  --output /tmp/minimax-h3-api-workflow.json \
  --width 864 --height 480 --duration 3
```
