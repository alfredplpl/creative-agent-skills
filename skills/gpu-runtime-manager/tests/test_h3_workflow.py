import json
import tempfile
import unittest
from pathlib import Path

from helpers import SCRIPTS  # noqa: F401
from gpu_runtime import ConfigurationError
from h3_workflow import build_h3_workflow, build_h3_workflow_from_prompt_file, write_workflow


def node(graph, class_type):
    matches = [item for item in graph.values() if item.get("class_type") == class_type]
    if len(matches) != 1:
        raise AssertionError(f"expected one {class_type}, found {len(matches)}")
    return matches[0]


class H3WorkflowTests(unittest.TestCase):
    def test_builds_workflow_from_prompt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.txt"
            prompt.write_text("anime runner\noverall_soundscape: footsteps", encoding="utf-8")
            graph = build_h3_workflow_from_prompt_file(
                prompt, width=864, height=480, duration=3.0, seed=123,
                output_prefix="video/test-run",
            )
        video = node(graph, "MiniMaxH3ImageToVideo")["inputs"]
        self.assertEqual(video["prompt"], "anime runner\noverall_soundscape: footsteps")
        self.assertEqual((video["width"], video["height"]), (864, 480))
        self.assertEqual(node(graph, "PrimitiveFloat")["inputs"]["value"], 3.0)
        noise = node(graph, "RandomNoise")["inputs"]
        self.assertEqual(noise["noise_seed"], 123)
        self.assertNotIn("value", noise)
        self.assertEqual(node(graph, "SaveVideo")["inputs"]["filename_prefix"],
                         "video/test-run")

    def test_random_seed_is_in_uint64_range(self):
        graph = build_h3_workflow("prompt")
        seed = node(graph, "RandomNoise")["inputs"]["noise_seed"]
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**64)

    def test_rejects_invalid_generation_values(self):
        invalid = [
            {"width": 850},
            {"height": 0},
            {"duration": 15.1},
            {"seed": -1},
            {"output_prefix": "../escape"},
        ]
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ConfigurationError):
                    build_h3_workflow("prompt", **options)

    def test_rejects_empty_prompt_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.txt"
            prompt.write_text("  \n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                build_h3_workflow_from_prompt_file(prompt)

    def test_write_workflow_outputs_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "workflow.json"
            graph = build_h3_workflow("prompt", seed=7)
            write_workflow(output, graph)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), graph)


if __name__ == "__main__":
    unittest.main()
