import pytest

from agentic_core.rag.chunking import chunk_document
from agentic_core.rag.parsing import UnsupportedDocument, parse_bytes, parse_text

from .helpers import make_docx, make_pdf


@pytest.mark.parametrize(
    "query, expected",
    [
        ("How many days of annual leave can I carry forward?", "§3.5"),
        ("Can I see a colleague's leave balance?", "§Q10"),
        ("gift limit", "HR-POL-003"),
        ("medical certificate for sick leave", "§4.1"),
        ("maternity leave weeks", "HR-POL-002"),
    ],
)
def test_policy_search_finds_relevant_passages(service, query, expected):
    hits = service.knowledge.search(query, scope="policy", top_k=3)
    assert any(expected in h.citation for h in hits), [h.citation for h in hits]


def test_broad_policy_question_returns_overview_first(service):
    hits = service.knowledge.search("What is our parental leave policy?", scope="policy", top_k=3)
    assert hits[0].doc_code == "HR-POL-002"
    assert hits[0].section == "overview"


def test_citations_carry_version_and_section(service):
    hit = service.knowledge.search("carry forward annual leave", scope="policy", top_k=1)[0]
    assert hit.version and hit.section
    assert hit.citation.startswith(hit.title)


def test_policy_index_is_idempotent(service):
    summary = service.knowledge.index_policies()
    assert summary["indexed"] == []
    assert len(summary["unchanged"]) == 5


def test_uploads_are_scoped_to_their_conversation(service):
    k = service.knowledge
    info = k.ingest_upload(conversation_id="c1", filename="travel.md", text="# Travel\n\n## Per diem\n\nDomestic per diem is INR 3,500.")
    assert info["status"] == "indexed"
    again = k.ingest_upload(conversation_id="c1", filename="travel.md", text="# Travel\n\n## Per diem\n\nDomestic per diem is INR 3,500.")
    assert again["status"] == "unchanged"
    assert k.search("per diem", scope="conversation", conversation_id="c1")
    assert k.search("per diem", scope="conversation", conversation_id="c2") == []
    assert k.search("per diem", scope="conversation", conversation_id=None) == []
    assert k.has_uploads("c1") and not k.has_uploads("c2")
    assert k.delete_document(info["id"])
    assert k.search("per diem", scope="conversation", conversation_id="c1") == []


def test_pdf_and_docx_uploads(service):
    pdf = make_pdf(["Relocation allowance is INR 50,000.", "It is paid with the first salary."])
    parsed = parse_bytes("relocation.pdf", pdf)
    assert "Relocation allowance" in parsed.text and parsed.pages
    info = service.knowledge.ingest_upload(conversation_id="c3", filename="relocation.pdf", data=pdf)
    hits = service.knowledge.search("relocation allowance amount", scope="conversation", conversation_id="c3")
    assert info["chunks"] >= 1 and hits and "p.1" in hits[0].citation

    docx_bytes = make_docx("Onboarding Guide", {"Laptop": "Laptops are issued on day one.", "Badge": "Collect your badge from reception."})
    service.knowledge.ingest_upload(conversation_id="c3", filename="onboarding.docx", data=docx_bytes)
    hits = service.knowledge.search("when is the laptop issued", scope="conversation", conversation_id="c3")
    assert hits[0].filename == "onboarding.docx" and "Laptop" in hits[0].citation + hits[0].heading


def test_unsupported_binary_is_rejected():
    with pytest.raises(UnsupportedDocument):
        parse_bytes("image.png", b"\x89PNG\r\n\x1a\n\x00\x00\xff\xfe")


def test_chunker_labels_clauses():
    doc = parse_text(
        "p.md",
        "---\ndoc_id: X-1\ntitle: Demo\n---\n# Demo\n\n## 3. Rules\n\n3.1 **First.** One.\n\n3.2 **Second.** Two.\n\n"
        "## 4. More\n\n4.1 Three.\n\n## 5. Last\n\n5.1 **Final.** Four.",
    )
    assert doc.metadata["doc_id"] == "X-1"
    chunks = chunk_document(doc, "Demo")
    labels = [c.section for c in chunks]
    assert labels == ["overview", "3.1-3.2", "4.1", "5.1"]
    assert "3. Rules: First: One." in chunks[0].text
