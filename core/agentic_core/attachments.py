"""Normalises incoming OpenAI-style messages and pulls out attached documents.

LibreChat's "Upload as Text" prepends extracted file text to the user message:

    Attached document(s):
    ```md# "file.pdf"
    <text>

    ---

    # "other.docx"
    <text>

    ```
    <user message>

Other OpenAI clients may send `{"type": "file", "file": {"filename", "file_data"}}`
content parts. Both are turned into ExtractedFile objects; the text is removed
from the prompt and indexed for on-demand retrieval instead.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from typing import Any

LIBRECHAT_PREFIX = "Attached document(s):\n```md"
_FILE_SPLIT = re.compile(r'(?:^|\n\n---\n\n)# "(.+?)"\n')
_FOOTER = re.compile(r"\n+---\n_Agents: .*?_\s*$", re.DOTALL)


@dataclass
class ExtractedFile:
    filename: str
    text: str | None = None
    data: bytes | None = None

    @property
    def sha(self) -> str:
        raw = self.data if self.data is not None else (self.text or "").encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def split_librechat_context(text: str) -> tuple[list[ExtractedFile], str]:
    if not text or not text.startswith(LIBRECHAT_PREFIX):
        return [], text
    start = len(LIBRECHAT_PREFIX)
    # Each file body ends with "\n", the block closes with "\n```" and the user text follows
    # after a "\n" separator. Pick the last closing fence that leaves balanced fences on
    # both sides (documents and questions may contain their own code blocks).
    candidates = [m.start() for m in re.finditer(r"\n\n```\n", text) if m.start() >= start]
    if text.endswith("\n\n```"):
        candidates.append(len(text) - 5)
    end = -1
    for pos in candidates:
        block, rest = text[start:pos], text[pos + 5 :]
        if block.count("```") % 2 == 0 and rest.count("```") % 2 == 0:
            end = pos
    if end < 0:
        return [], text
    block = text[start : end + 1]
    user_text = text[end + 6 :] if end + 5 < len(text) else ""
    pieces = _FILE_SPLIT.split(block)
    files = []
    for i in range(1, len(pieces) - 1, 2):
        name = pieces[i].strip() or "attachment.txt"
        body = pieces[i + 1].strip("\n")
        if body.strip():
            files.append(ExtractedFile(filename=name, text=body))
    return files, user_text


def _decode_data_url(value: str) -> bytes | None:
    if not value:
        return None
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        return base64.b64decode(value, validate=False)
    except (ValueError, TypeError):
        return None


def content_to_text(content: Any) -> tuple[str, list[ExtractedFile], int]:
    """Returns (text, files, number of ignored image parts)."""
    if content is None:
        return "", [], 0
    if isinstance(content, str):
        files, text = split_librechat_context(content)
        return text, files, 0
    texts: list[str] = []
    files: list[ExtractedFile] = []
    images = 0
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text"):
            found, text = split_librechat_context(part.get("text") or "")
            files.extend(found)
            texts.append(text)
        elif kind in ("file", "input_file"):
            payload = part.get("file") if kind == "file" else part
            payload = payload or {}
            data = _decode_data_url(payload.get("file_data") or "")
            if data:
                files.append(ExtractedFile(filename=payload.get("filename") or "attachment", data=data))
        elif kind in ("image_url", "input_image", "image"):
            images += 1
    return "\n".join(t for t in texts if t), files, images


@dataclass
class NormalizedConversation:
    history: list[dict[str, str]]
    user_message: str
    files: list[ExtractedFile]
    system_notes: list[str]
    ignored_images: int
    first_user_message: str


def normalize_messages(messages: list[dict[str, Any]]) -> NormalizedConversation:
    turns: list[dict[str, str]] = []
    files: dict[str, ExtractedFile] = {}
    notes: list[str] = []
    images = 0
    for msg in messages:
        role = msg.get("role")
        text, found, image_count = content_to_text(msg.get("content"))
        if role == "system" or role == "developer":
            if text.strip():
                notes.append(text.strip())
            continue
        if role not in ("user", "assistant"):
            continue
        for f in found:
            files.setdefault(f.sha, f)
        if role == "user":
            images += image_count
        if role == "assistant":
            text = _FOOTER.sub("", text)
        turns.append({"role": role, "content": text.strip()})
    user_message = ""
    if turns and turns[-1]["role"] == "user":
        user_message = turns.pop()["content"]
    first_user = next((t["content"] for t in turns if t["role"] == "user"), user_message)
    history = [t for t in turns if t["content"]]
    return NormalizedConversation(
        history=history,
        user_message=user_message,
        files=list(files.values()),
        system_notes=notes,
        ignored_images=images,
        first_user_message=first_user,
    )
