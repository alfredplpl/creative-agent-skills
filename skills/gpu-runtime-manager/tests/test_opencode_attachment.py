import base64
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from helpers import SCRIPTS  # noqa: F401
from opencode_attachment import (OpenCodeAttachmentError,
                                 cleanup_staged_attachment,
                                 resolve_opencode_attachment)


class OpenCodeAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.database = self.root / "opencode.db"
        self.connection = sqlite3.connect(self.database)
        self.connection.executescript(
            "CREATE TABLE session ("
            "id TEXT PRIMARY KEY, directory TEXT NOT NULL, "
            "time_updated INTEGER NOT NULL, time_archived INTEGER);"
            "CREATE TABLE message (id TEXT PRIMARY KEY, data TEXT NOT NULL);"
            "CREATE TABLE part (message_id TEXT NOT NULL, session_id TEXT NOT NULL, "
            "time_created INTEGER NOT NULL, data TEXT NOT NULL);"
        )

    def tearDown(self):
        self.connection.close()
        self.temp.cleanup()

    def add_session(self, session_id="session-new", updated=10_000):
        self.connection.execute(
            "INSERT INTO session VALUES (?, ?, ?, NULL)",
            (session_id, str(self.project), updated),
        )
        self.connection.commit()

    def add_image(self, content=b"attached-image", *, filename="character.png",
                  session_id="session-new", created=10_001, mime="image/png"):
        message_id = f"message-{created}"
        self.connection.execute(
            "INSERT INTO message VALUES (?, ?)",
            (message_id, json.dumps({"role": "user"})),
        )
        part = {
            "type": "file",
            "mime": mime,
            "filename": filename,
            "url": f"data:{mime};base64,{base64.b64encode(content).decode()}",
            "source": {"path": "/missing/original.png", "type": "file"},
        }
        self.connection.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?)",
            (message_id, session_id, created, json.dumps(part)),
        )
        self.connection.commit()

    def test_stages_embedded_attachment_without_original_file(self):
        self.add_session()
        self.add_image()
        path, metadata = resolve_opencode_attachment(
            "character.png", database=self.database, directory=self.project,
            staging_root=self.root / "staging",
        )
        self.assertEqual(path.read_bytes(), b"attached-image")
        self.assertEqual(metadata["filename"], "character.png")
        self.assertEqual(metadata["session_id"], "session-new")
        cleanup_staged_attachment(path)
        self.assertFalse(path.exists())

    def test_uses_latest_image_in_latest_project_session(self):
        self.add_session("session-old", 1_000)
        self.add_image(b"old", session_id="session-old", created=1_001)
        self.add_session("session-new", 10_000)
        self.add_image(b"first", created=10_001, filename="first.png")
        self.add_image(b"latest", created=10_002, filename="latest.png")
        path, _ = resolve_opencode_attachment(
            database=self.database, directory=self.project,
            staging_root=self.root / "staging",
        )
        self.assertEqual(path.read_bytes(), b"latest")

    def test_requires_session_when_project_sessions_are_simultaneously_active(self):
        self.add_session("session-a", 10_000)
        self.add_session("session-b", 9_000)
        self.add_image(session_id="session-a", created=10_001)
        with self.assertRaises(OpenCodeAttachmentError):
            resolve_opencode_attachment(
                database=self.database, directory=self.project,
                staging_root=self.root / "staging",
            )
        path, metadata = resolve_opencode_attachment(
            database=self.database, directory=self.project, session_id="session-a",
            staging_root=self.root / "staging",
        )
        self.assertTrue(path.is_file())
        self.assertEqual(metadata["session_id"], "session-a")

    def test_does_not_fall_back_to_an_older_session_attachment(self):
        self.add_session("session-old", 1_000)
        self.add_image(session_id="session-old", created=1_001)
        self.add_session("session-new", 10_000)
        with self.assertRaises(OpenCodeAttachmentError):
            resolve_opencode_attachment(
                "character.png", database=self.database, directory=self.project,
                staging_root=self.root / "staging",
            )

    def test_rejects_missing_filename_and_non_image_attachment(self):
        self.add_session()
        self.add_image(filename="other.png")
        with self.assertRaises(OpenCodeAttachmentError):
            resolve_opencode_attachment(
                "missing.png", database=self.database, directory=self.project,
                staging_root=self.root / "staging",
            )
        with self.assertRaises(OpenCodeAttachmentError):
            resolve_opencode_attachment(
                "../other.png", database=self.database, directory=self.project,
                staging_root=self.root / "staging",
            )


if __name__ == "__main__":
    unittest.main()
