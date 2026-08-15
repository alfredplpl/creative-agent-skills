import unittest

from helpers import SCRIPTS  # noqa: F401
from gpu_runtime import ConfigurationError, VRAMTimeout, wait_for_vram


class WaitVRAMTests(unittest.TestCase):
    def test_success_after_polling(self):
        values = iter([100, 500, 900])
        result = wait_for_vram(0, 800, 1, 0.001,
                               lambda _: {"free_vram_mb": next(values)})
        self.assertEqual(result["free_vram_mb"], 900)

    def test_timeout(self):
        with self.assertRaises(VRAMTimeout):
            wait_for_vram(0, 800, 0.005, 0.001, lambda _: {"free_vram_mb": 100})

    def test_invalid_arguments(self):
        with self.assertRaises(ConfigurationError):
            wait_for_vram(0, 0, 1, 0.1, lambda _: {})


if __name__ == "__main__":
    unittest.main()
