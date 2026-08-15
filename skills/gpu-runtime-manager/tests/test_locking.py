import tempfile
import unittest
from pathlib import Path

from helpers import SCRIPTS  # noqa: F401
from gpu_runtime import FileLock, LockTimeout


class LockTests(unittest.TestCase):
    def test_second_transition_times_out(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "manager.lock")
            with FileLock(path, 0.1):
                with self.assertRaises(LockTimeout):
                    with FileLock(path, 0.01):
                        pass


if __name__ == "__main__":
    unittest.main()
