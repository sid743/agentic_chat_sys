import base64

from agentic_core.attachments import content_to_text, normalize_messages, split_librechat_context


def librechat_message(files: dict[str, str], question: str) -> str:
    """Exactly how LibreChat (packages/api/src/files/context.ts) prepends 'Upload as Text' files."""
    text = ""
    for name, body in files.items():
        text += ("Attached document(s):\n```md" if not text else "\n\n---\n\n") + f'# "{name}"\n{body}\n'
    text += "\n```"
    return f"{text}\n{question}"


def test_split_single_and_multiple_files():
    msg = librechat_message({"a.pdf": "Alpha text", "b.docx": "Beta\n\nsecond paragraph"}, "Compare them")
    files, question = split_librechat_context(msg)
    assert question == "Compare them"
    assert [f.filename for f in files] == ["a.pdf", "b.docx"]
    assert files[0].text == "Alpha text"
    assert files[1].text == "Beta\n\nsecond paragraph"


def test_code_fences_inside_documents_and_question():
    body = "Config example:\n```yaml\nkey: value\n```\nEnd of doc."
    msg = librechat_message({"guide.md": body}, "Explain this snippet:\n```\nprint(1)\n```")
    files, question = split_librechat_context(msg)
    assert files[0].text == body
    assert question == "Explain this snippet:\n```\nprint(1)\n```"


def test_plain_messages_untouched():
    files, question = split_librechat_context("Just a question")
    assert files == [] and question == "Just a question"


def test_openai_file_parts_and_images():
    data = base64.b64encode(b"hello file").decode()
    text, files, images = content_to_text(
        [
            {"type": "text", "text": "What is in the file?"},
            {"type": "file", "file": {"filename": "note.txt", "file_data": f"data:text/plain;base64,{data}"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
        ]
    )
    assert text == "What is in the file?"
    assert files[0].filename == "note.txt" and files[0].data == b"hello file"
    assert images == 1


def test_normalize_history_dedupes_files_and_strips_footer():
    first = librechat_message({"policy.md": "Per diem is 3500"}, "What is the per diem?")
    convo = normalize_messages(
        [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": first},
            {"role": "assistant", "content": "It is 3500.\n\n---\n_Agents: Document Agent · model `x` · run `r`_"},
            {"role": "user", "content": first.replace("What is the per diem?", "And flights?")},
        ]
    )
    assert convo.user_message == "And flights?"
    assert convo.first_user_message == "What is the per diem?"
    assert len(convo.files) == 1
    assert convo.history[-1] == {"role": "assistant", "content": "It is 3500."}
    assert convo.system_notes == ["Be brief"]
