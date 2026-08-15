import tempfile
import unittest
from unittest import mock

from helpers import FakeEnvironment, config
from gpu_runtime import (AtomicRecoveryError, ComfyOperationError, ComfyUnavailable, GPUOwner,
                         LlamaOperationError, LlamaUnavailable, RuntimeManager,
                         UnexpectedGPUProcess, UnknownGPUState, VRAMTimeout, WorkflowBusy,
                         WorkflowExecutionError, WorkflowSubmissionError)


class RuntimeManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = FakeEnvironment()
        self.manager = RuntimeManager(config(self.temp.name), self.env.monitor, self.env.http)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def workflow():
        return {"1": {"class_type": "TestVideoNode", "inputs": {"prompt": "run"}}}

    def test_none_to_llm(self):
        result = self.manager.acquire(GPUOwner.LLM)
        self.assertEqual(result["gpu_owner"], "llm")
        self.assertEqual(self.env.llama_state, "loaded")

    def test_none_to_video_reserves_owner(self):
        result = self.manager.acquire(GPUOwner.VIDEO)
        self.assertEqual(result["gpu_owner"], "video")
        self.assertEqual(result["detected_state"], "none")

    def test_llm_to_video(self):
        self.manager.acquire(GPUOwner.LLM)
        result = self.manager.acquire(GPUOwner.VIDEO)
        self.assertEqual(result["gpu_owner"], "video")
        self.assertEqual(self.env.llama_state, "unloaded")

    def test_llm_to_video_waits_for_async_router_unload(self):
        self.manager.acquire(GPUOwner.LLM)
        self.env.unload_polls_remaining = 3
        result = self.manager.acquire(GPUOwner.VIDEO)
        self.assertEqual(result["gpu_owner"], "video")
        self.assertEqual(self.env.llama_state, "unloaded")

    def test_video_to_llm(self):
        self.manager.acquire(GPUOwner.VIDEO)
        self.env.video_loaded = True
        result = self.manager.acquire(GPUOwner.LLM)
        self.assertEqual(result["gpu_owner"], "llm")
        self.assertFalse(self.env.video_loaded)

    def test_idempotent_llm(self):
        self.manager.acquire(GPUOwner.LLM)
        load_count = len([call for call in self.env.calls if call[1] == "/models/load"])
        self.manager.acquire(GPUOwner.LLM)
        self.assertEqual(load_count, len([call for call in self.env.calls if call[1] == "/models/load"]))

    def test_idempotent_video(self):
        self.manager.acquire(GPUOwner.VIDEO)
        calls = len(self.env.calls)
        self.manager.acquire(GPUOwner.VIDEO)
        self.assertGreaterEqual(len(self.env.calls), calls)
        self.assertFalse(any(call[1] == "/free" for call in self.env.calls))

    def test_release_llm_and_video(self):
        self.manager.acquire(GPUOwner.LLM)
        self.assertEqual(self.manager.release(GPUOwner.LLM)["gpu_owner"], "none")
        self.manager.acquire(GPUOwner.VIDEO); self.env.video_loaded = True
        self.assertEqual(self.manager.release(GPUOwner.VIDEO)["gpu_owner"], "none")

    def test_vram_timeout_after_comfy_free(self):
        self.manager.acquire(GPUOwner.VIDEO)
        self.env.video_loaded = True; self.env.retain_video_memory = True
        with self.assertRaises(VRAMTimeout): self.manager.acquire(GPUOwner.LLM)

    def test_llama_unreachable(self):
        self.env.llama_reachable = False
        with self.assertRaises(LlamaUnavailable): self.manager.acquire(GPUOwner.LLM)

    def test_llama_load_failure(self):
        self.env.load_fails = True
        with self.assertRaises(LlamaOperationError): self.manager.acquire(GPUOwner.LLM)

    def test_llama_wake_failure(self):
        self.env.llama_state = "sleeping"; self.env.load_fails = True
        with self.assertRaises(LlamaOperationError): self.manager.acquire(GPUOwner.LLM)

    def test_llama_unload_failure(self):
        self.manager.acquire(GPUOwner.LLM); self.env.unload_fails = True
        with self.assertRaises(LlamaOperationError): self.manager.acquire(GPUOwner.VIDEO)

    def test_comfy_unreachable(self):
        self.env.comfy_reachable = False
        with self.assertRaises(ComfyUnavailable): self.manager.acquire(GPUOwner.VIDEO)

    def test_comfy_busy_timeout(self):
        self.manager.acquire(GPUOwner.VIDEO); self.env.video_loaded = True; self.env.comfy_running = 1
        with self.assertRaises(WorkflowBusy): self.manager.acquire(GPUOwner.LLM)

    def test_comfy_free_failure(self):
        self.manager.acquire(GPUOwner.VIDEO); self.env.video_loaded = True; self.env.free_fails = True
        with self.assertRaises(ComfyOperationError): self.manager.acquire(GPUOwner.LLM)

    def test_unknown_process_stops_acquire(self):
        self.env.rogue_process = True
        with self.assertRaises(UnexpectedGPUProcess): self.manager.acquire(GPUOwner.LLM)

    def test_conflicting_runtime_state_is_unknown(self):
        self.env.llama_state = "loaded"; self.env.video_loaded = True
        self.assertEqual(self.manager.snapshot()["detected_state"], "unknown")
        with self.assertRaises(UnknownGPUState): self.manager.acquire(GPUOwner.LLM)

    def test_dry_run_does_not_mutate(self):
        result = self.manager.acquire(GPUOwner.LLM, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(self.env.llama_state, "unloaded")

    def test_atomic_video_restores_llm_after_success(self):
        self.manager.acquire(GPUOwner.LLM)
        result = self.manager.run_video_workflow(self.workflow())
        self.assertEqual(result["gpu_owner"], "llm")
        self.assertEqual(result["atomic_video"]["status"], "success")
        self.assertEqual(result["atomic_video"]["files"][0]["filename"], "result.mp4")
        self.assertEqual(self.env.submitted_workflow, {"prompt": self.workflow()})
        paths = [call[1] for call in self.env.calls]
        self.assertLess(paths.index("/models/unload"), paths.index("/prompt"))
        self.assertLess(paths.index("/prompt"), paths.index("/free"))
        self.assertLess(paths.index("/free"), len(paths) - 1 - paths[::-1].index("/models/load"))

    def test_atomic_video_restores_llm_after_submission_failure(self):
        self.manager.acquire(GPUOwner.LLM)
        self.env.workflow_submit_fails = True
        with self.assertRaises(WorkflowSubmissionError):
            self.manager.run_video_workflow(self.workflow())
        self.assertEqual(self.env.llama_state, "loaded")
        self.assertFalse(self.env.video_loaded)

    def test_atomic_video_restores_llm_after_node_validation_failure(self):
        self.manager.acquire(GPUOwner.LLM)
        self.env.workflow_node_errors = True
        with self.assertRaises(WorkflowSubmissionError):
            self.manager.run_video_workflow(self.workflow())
        self.assertEqual(self.env.llama_state, "loaded")
        self.assertFalse(self.env.video_loaded)

    def test_atomic_video_restores_llm_after_execution_failure(self):
        self.manager.acquire(GPUOwner.LLM)
        self.env.workflow_status = "error"
        with self.assertRaises(WorkflowExecutionError):
            self.manager.run_video_workflow(self.workflow())
        self.assertEqual(self.env.llama_state, "loaded")
        self.assertFalse(self.env.video_loaded)

    def test_atomic_video_does_not_restore_llm_while_workflow_is_busy(self):
        self.manager.acquire(GPUOwner.LLM)
        self.env.workflow_never_completes = True
        with self.assertRaises(AtomicRecoveryError):
            self.manager.run_video_workflow(self.workflow(), timeout=0.002)
        self.assertEqual(self.env.llama_state, "unloaded")
        self.assertEqual(self.env.comfy_running, 1)

    def test_atomic_video_dry_run_does_not_submit_or_unload(self):
        self.manager.acquire(GPUOwner.LLM)
        result = self.manager.run_video_workflow(self.workflow(), dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(self.env.llama_state, "loaded")
        self.assertNotIn("/prompt", [call[1] for call in self.env.calls])

    def test_atomic_video_requires_llm_owner(self):
        with self.assertRaises(UnknownGPUState):
            self.manager.run_video_workflow(self.workflow())

    def test_atomic_video_rejects_stale_persisted_llm_owner(self):
        self.manager._persist(GPUOwner.LLM)
        with self.assertRaises(UnknownGPUState):
            self.manager.run_video_workflow(self.workflow())

    def test_reference_upload_happens_only_from_measured_llm_state(self):
        with mock.patch("gpu_runtime.upload_comfy_image") as upload:
            with self.assertRaises(UnknownGPUState):
                self.manager.upload_reference_image("character.png")
            upload.assert_not_called()

            self.manager.acquire(GPUOwner.LLM)
            upload.return_value = "gpu-runtime-manager/reference.png"
            result = self.manager.upload_reference_image("character.png")

        self.assertEqual(result, "gpu-runtime-manager/reference.png")
        self.assertEqual(self.env.llama_state, "loaded")
        self.assertNotEqual(self.env.calls[-1][1], "/models/unload")


if __name__ == "__main__":
    unittest.main()
