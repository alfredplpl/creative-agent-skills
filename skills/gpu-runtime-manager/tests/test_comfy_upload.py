import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import config
from gpu_runtime import ComfyUploadError, upload_comfy_image


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.payload


class ComfyUploadTests(unittest.TestCase):
    def test_uploads_multipart_image_and_returns_load_image_name(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "character.png"
            image.write_bytes(b"fake-png-data")
            captured = {}

            def open_request(request, timeout):
                captured.update(request=request, timeout=timeout)
                return FakeResponse({
                    "name": "reference-abc.png",
                    "subfolder": "gpu-runtime-manager",
                    "type": "input",
                })

            with mock.patch("gpu_runtime.urllib.request.urlopen", open_request):
                result = upload_comfy_image(config(directory), image)

        self.assertEqual(result, "gpu-runtime-manager/reference-abc.png")
        self.assertTrue(captured["request"].full_url.endswith("/upload/image"))
        self.assertIn(b'form-data; name="image"', captured["request"].data)
        self.assertIn(b"fake-png-data", captured["request"].data)
        self.assertEqual(captured["timeout"], 30)

    def test_rejects_missing_empty_and_unsafe_upload_response(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            with self.assertRaises(ComfyUploadError):
                upload_comfy_image(cfg, Path(directory) / "missing.png")

            image = Path(directory) / "empty.png"
            image.write_bytes(b"")
            with self.assertRaises(ComfyUploadError):
                upload_comfy_image(cfg, image)

            image.write_bytes(b"data")
            response = FakeResponse({"name": "escape.png", "subfolder": "../outside"})
            with mock.patch("gpu_runtime.urllib.request.urlopen", return_value=response):
                with self.assertRaises(ComfyUploadError):
                    upload_comfy_image(cfg, image)

            unsupported = Path(directory) / "reference.txt"
            unsupported.write_bytes(b"not-an-image")
            with self.assertRaises(ComfyUploadError):
                upload_comfy_image(cfg, unsupported)


if __name__ == "__main__":
    unittest.main()
