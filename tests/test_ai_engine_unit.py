import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

# Ensure the project root is importable when running tests directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import ai_engine


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    async def ainvoke(self, _query):
        return self.docs


class FakeStore:
    def __init__(self, docs=None):
        self.docs = docs or []

    def add_documents(self, docs):
        self.docs.extend(docs)

    def as_retriever(self, search_kwargs=None):
        return FakeRetriever(self.docs)


class FakeLLM:
    def __init__(self, content="response"):
        self.content = content

    async def ainvoke(self, _messages):
        return SimpleNamespace(content=self.content)


class FakeSplitter:
    def __init__(self, *args, **kwargs):
        pass

    def split_documents(self, docs):
        return docs


class FakePDFLoader:
    def __init__(self, _path):
        pass

    def load_and_split(self):
        return [
            Document(page_content="page one"),
            Document(page_content="page two"),
        ]


@pytest.fixture(autouse=True)
def restore_globals(monkeypatch, tmp_path):
    """Isolate filesystem and vector store globals for each test."""
    monkeypatch.setattr(ai_engine, "UPLOAD_DIR", str(tmp_path / "uploads"), False)
    monkeypatch.setattr(ai_engine, "CHROMA_DB_DIR", str(tmp_path / "chroma"), False)
    yield


def test_save_file_locally(monkeypatch, tmp_path):
    buf = io.BytesIO(b"hello world")
    path = ai_engine.save_file_locally(buf, "hello.txt", trial_id=1)
    assert os.path.exists(path)
    with open(path, "rb") as fh:
        assert fh.read() == b"hello world"


@pytest.mark.asyncio
async def test_process_file_to_text_pdf(monkeypatch):
    monkeypatch.setattr(ai_engine, "PyPDFLoader", FakePDFLoader)
    text = await ai_engine.process_file_to_text("dummy.pdf", "application/pdf")
    assert "page one" in text and "page two" in text


@pytest.mark.asyncio
async def test_process_file_to_text_image(monkeypatch, tmp_path):
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"fakeimagebytes")
    monkeypatch.setattr(
        ai_engine, "ChatOpenAI", lambda *a, **k: FakeLLM("vision reply")
    )
    text = await ai_engine.process_file_to_text(str(img_path), "image/jpeg")
    assert "[Image Description" in text
    assert "vision reply" in text


def test_ingest_text_evidence(monkeypatch):
    fake_store = FakeStore()
    monkeypatch.setattr(ai_engine, "get_trial_vector_store", lambda _tid: fake_store)
    monkeypatch.setattr(ai_engine, "RecursiveCharacterTextSplitter", FakeSplitter)

    ai_engine.ingest_text_evidence(1, "case background text", "Case Background")

    assert fake_store.docs, "Documents should be added to the store"
    assert fake_store.docs[0].metadata["source"] == "Case Background"


@pytest.mark.asyncio
async def test_retrieve_context_and_sources(monkeypatch):
    docs = [
        Document(page_content="fact A", metadata={"source": "file1.pdf"}),
        Document(page_content="fact B", metadata={"source": "file1.pdf"}),
    ]
    monkeypatch.setattr(
        ai_engine, "get_trial_vector_store", lambda _tid: FakeStore(docs)
    )

    context, sources = await ai_engine.retrieve_context_and_sources(1, "query")

    assert "fact A" in context and "fact B" in context
    assert sources == ["file1.pdf"], "Sources should be de-duplicated"


@pytest.mark.asyncio
async def test_batch_generate_initial_arguments(monkeypatch):
    async def fake_generate(tid, bg, arg):
        return {"content": f"reply to {arg}", "sources": ["fileA"]}

    monkeypatch.setattr(ai_engine, "_generate_single_argument_with_rag", fake_generate)

    responses = await ai_engine.batch_generate_initial_arguments(1, "bg", ["a", "b"])

    assert len(responses) == 2
    assert responses[0]["content"] == "reply to a"


@pytest.mark.asyncio
async def test_generate_reply_with_new_evidence(monkeypatch):
    async def fake_retrieve(tid, query):
        return "ctx", ["s1"]

    monkeypatch.setattr(ai_engine, "retrieve_context_and_sources", fake_retrieve)
    monkeypatch.setattr(ai_engine, "ChatOpenAI", lambda *a, **k: FakeLLM("ai reply"))

    result = await ai_engine.generate_reply_with_new_evidence(
        trial_id=1,
        case_background="case bg",
        user_text="hello",
        chat_history=[],
        new_file_path=None,
        new_file_type=None,
    )

    assert result == "ai reply"
