#!/usr/bin/env python3
"""Resolve an OpenCode user image attachment without searching the filesystem."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import unquote_to_bytes

from gpu_runtime import RuntimeManagerError, validate_reference_image

MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024


class OpenCodeAttachmentError(RuntimeManagerError):
    code = "opencode_attachment_failure"


def default_opencode_db() -> Path:
    configured = os.environ.get("OPENCODE_DB")
    if configured:
        return Path(configured).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return root / "opencode" / "opencode.db"


def _read_data_url(url: str, expected_mime: str) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
        raise OpenCodeAttachmentError("OpenCode attachment has no embedded data URL")
    header, payload = url[5:].split(",", 1)
    fields = header.split(";")
    mime = fields[0].lower()
    if not mime.startswith("image/") or mime != expected_mime.lower():
        raise OpenCodeAttachmentError(
            f"OpenCode attachment MIME mismatch: part={expected_mime!r} data={mime!r}"
        )
    try:
        content = (base64.b64decode(payload, validate=True)
                   if "base64" in fields[1:] else unquote_to_bytes(payload))
    except (ValueError, binascii.Error) as exc:
        raise OpenCodeAttachmentError(f"invalid OpenCode attachment data: {exc}") from exc
    if not content:
        raise OpenCodeAttachmentError("OpenCode attachment is empty")
    if len(content) > MAX_ATTACHMENT_BYTES:
        raise OpenCodeAttachmentError(
            f"OpenCode attachment exceeds {MAX_ATTACHMENT_BYTES // 1048576} MB"
        )
    return content


def _active_session(connection: sqlite3.Connection, directory: Path,
                    session_id: str | None) -> str:
    if session_id:
        row = connection.execute(
            "SELECT id, directory FROM session WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise OpenCodeAttachmentError(f"OpenCode session not found: {session_id}")
        if Path(row[1]).resolve() != directory:
            raise OpenCodeAttachmentError(
                f"OpenCode session belongs to another directory: {row[1]}"
            )
        return str(row[0])

    rows = connection.execute(
        "SELECT id, time_updated FROM session "
        "WHERE directory = ? AND time_archived IS NULL "
        "ORDER BY time_updated DESC LIMIT 2",
        (str(directory),),
    ).fetchall()
    if not rows:
        raise OpenCodeAttachmentError(
            f"no active OpenCode session found for {directory}"
        )
    if len(rows) > 1 and int(rows[0][1]) - int(rows[1][1]) <= 2000:
        raise OpenCodeAttachmentError(
            "multiple OpenCode sessions are active for this directory; "
            "pass --opencode-session"
        )
    return str(rows[0][0])


def _image_part(connection: sqlite3.Connection, session_id: str,
                filename: str | None) -> dict:
    rows = connection.execute(
        "SELECT p.data, m.data FROM part p "
        "JOIN message m ON m.id = p.message_id "
        "WHERE p.session_id = ? ORDER BY p.time_created DESC",
        (session_id,),
    )
    for part_raw, message_raw in rows:
        try:
            part, message = json.loads(part_raw), json.loads(message_raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if message.get("role") != "user" or part.get("type") != "file":
            continue
        if not str(part.get("mime", "")).lower().startswith("image/"):
            continue
        if filename is not None and part.get("filename") != filename:
            continue
        return part
    wanted = filename if filename is not None else "the latest image"
    raise OpenCodeAttachmentError(
        f"OpenCode user attachment {wanted!r} was not found in session {session_id}"
    )


def resolve_opencode_attachment(
    filename: str | None = None,
    *,
    session_id: str | None = None,
    database: str | Path | None = None,
    directory: str | Path | None = None,
    staging_root: str | Path = "/tmp/gpu-runtime-manager-opencode",
) -> tuple[Path, dict[str, str]]:
    """Stage the selected attachment data and return (path, non-secret metadata)."""
    if filename is not None and (not filename or Path(filename).name != filename):
        raise OpenCodeAttachmentError("attachment filename must not contain a path")
    db = Path(database) if database else default_opencode_db()
    working_directory = Path(directory or Path.cwd()).resolve()
    if not db.is_file():
        raise OpenCodeAttachmentError(f"OpenCode database not found: {db}")
    try:
        connection = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
        with connection:
            selected_session = _active_session(connection, working_directory, session_id)
            part = _image_part(connection, selected_session, filename)
    except sqlite3.Error as exc:
        raise OpenCodeAttachmentError(f"cannot read OpenCode attachments: {exc}") from exc
    finally:
        if "connection" in locals():
            connection.close()

    content = _read_data_url(part.get("url"), str(part.get("mime", "")))
    original = str(part.get("filename", "attachment.png"))
    suffix = Path(original).suffix.lower()
    digest = hashlib.sha256(content).hexdigest()
    session_key = hashlib.sha256(selected_session.encode()).hexdigest()[:16]
    target_dir = Path(staging_root) / session_key
    invocation = f"{os.getpid()}-{os.urandom(6).hex()}"
    target = target_dir / f"{digest}-{invocation}{suffix}"
    try:
        target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.chmod(0o600)
        os.replace(temporary, target)
    except OSError as exc:
        raise OpenCodeAttachmentError(f"cannot stage OpenCode attachment: {exc}") from exc
    try:
        validate_reference_image(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target, {
        "filename": original,
        "session_id": selected_session,
        "sha256": digest,
    }


def cleanup_staged_attachment(path: str | Path) -> None:
    target = Path(path)
    target.unlink(missing_ok=True)
    try:
        target.parent.rmdir()
    except OSError:
        pass
