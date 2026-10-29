"""Chunking, a local persistent vector store, and a patient-filtered retriever for the chart corpus.

Every retrieval fix (see `notebooks/optional/improve_retrieval.ipynb`) is a field on `RetrievalConfig`, so a single object describes a pipeline variant:

| Field              | naive baseline | What the fix does                                                  |
|--------------------|----------------|--------------------------------------------------------------------|
| `chunker`          | "naive"        | "structure": split on sections, keep tables whole, add a header    |
| `chunk_size`       | 500            | bigger chunks keep addenda with their context                      |
| `chunk_overlap`    | 0              | overlap stops a key sentence from being cut off its subject        |
| `date_aware`       | False          | show each source's date and return results newest first            |
| `query_rewrite`    | False          | rewrite the question into clinical shorthand (SOB, c/o, NC…)       |
| `bind_patient`     | False          | always filter on the session's patient ID, never the model's guess |

Vector store: Chroma persisted to `.chroma/` (one collection per chunking config and embeddings
model). If Chroma can't be imported or started, it falls back to LangChain's `InMemoryVectorStore`
automatically, so the lab still runs.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import warnings
from dataclasses import asdict, dataclass, replace

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.corpus import ROOT, ChartDoc, load_documents, slugify
from utils.models import embeddings_id, get_embeddings, get_model

CHROMA_DIR = ROOT / ".chroma"


@dataclass(frozen=True)
class RetrievalConfig:
    name: str = "baseline"
    chunker: str = "naive"  # "naive" | "structure"
    chunk_size: int = 500
    chunk_overlap: int = 0
    k: int = 4
    date_aware: bool = False
    query_rewrite: bool = False
    bind_patient: bool = False

    def chunk_key(self) -> str:
        return f"{self.chunker}-{self.chunk_size}-{self.chunk_overlap}"

    def with_(self, **changes) -> "RetrievalConfig":
        return replace(self, **changes)


BASELINE = RetrievalConfig()
# v1: structure-aware chunking (tables stay whole) + date-aware ranking + session-bound patient filter.
IMPROVED_V1 = RetrievalConfig(
    name="improved-v1", chunker="structure", chunk_size=800, chunk_overlap=0, k=5,
    date_aware=True, bind_patient=True,
)
# v2: v1 + abbreviation-aware query rewriting + bigger chunks with overlap (addenda keep their context).
IMPROVED_V2 = IMPROVED_V1.with_(name="improved-v2", chunk_size=2400, chunk_overlap=300, k=6, query_rewrite=True)
CONFIGS = {c.name: c for c in (BASELINE, IMPROVED_V1, IMPROVED_V2)}


# --------------------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------------------

def _section_at(doc: ChartDoc, offset: int) -> str:
    """Anchor of the `## ` section that contains character `offset` of the body."""
    anchor = "header"
    for m in re.finditer(r"(?m)^## (.+)$", doc.body):
        if m.start() > offset:
            break
        anchor = slugify(m.group(1))
    return anchor


def naive_chunks(docs, chunk_size: int = 500, chunk_overlap: int = 0) -> list[Document]:
    """Lab 1 default: fixed-size character chunks over the raw body. No awareness of tables or sections,
    and the chunk text carries no date or document context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    out = []
    for doc in docs:
        for i, piece in enumerate(splitter.create_documents([doc.body], metadatas=[doc.metadata()])):
            start = piece.metadata.pop("start_index", 0)
            piece.metadata.update({"section": _section_at(doc, max(start, 0)), "chunk": i})
            out.append(piece)
    return out


def _blocks(text: str) -> list[str]:
    """Split a section into paragraphs, keeping every Markdown table as ONE atomic block."""
    blocks, buf, in_table = [], [], False
    for line in text.splitlines():
        is_table = line.lstrip().startswith("|")
        if is_table != in_table and buf:
            blocks.append("\n".join(buf).strip())
            buf = []
        in_table = is_table
        if not is_table and not line.strip():
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def structure_chunks(docs, chunk_size: int = 800, chunk_overlap: int = 0) -> list[Document]:
    """Section-aware chunking. Each chunk:
    - stays inside one `## ` section,
    - never splits a Markdown table (a table is atomic, even if longer than `chunk_size`),
    - starts with a context header: title | doc type | date | section.
    """
    fallback = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for doc in docs:
        n = 0
        for sec in doc.sections:
            header = f"[{doc.title} | {doc.doc_type} | {doc.date} | section: {sec.heading}]"
            pieces: list[str] = []
            current = ""
            for block in _blocks(sec.text):
                if block.startswith("|"):
                    if current:
                        pieces.append(current)
                        current = ""
                    pieces.append(block)
                    continue
                if len(current) + len(block) + 2 <= chunk_size:
                    current = f"{current}\n\n{block}" if current else block
                    continue
                if current:
                    pieces.append(current)
                if len(block) > chunk_size:
                    pieces.extend(fallback.split_text(block))
                    current = ""
                else:
                    tail = current[-chunk_overlap:] if chunk_overlap and current else ""
                    current = f"{tail}\n\n{block}".strip() if tail else block
            if current:
                pieces.append(current)
            for piece in pieces:
                meta = {**doc.metadata(), "section": sec.anchor, "chunk": n}
                out.append(Document(page_content=f"{header}\n{piece}", metadata=meta))
                n += 1
    return out


def chunk_corpus(config: RetrievalConfig = BASELINE, docs=None) -> list[Document]:
    docs = list(docs or load_documents())
    if config.chunker == "structure":
        return structure_chunks(docs, config.chunk_size, config.chunk_overlap)
    return naive_chunks(docs, config.chunk_size, config.chunk_overlap)


# --------------------------------------------------------------------------------------------------
# Vector store (Chroma, persistent) with in-memory fallback
# --------------------------------------------------------------------------------------------------

_STORES: dict[str, object] = {}
_BUILD_LOCK = threading.Lock()  # agents call tools in parallel threads; build/open each index exactly once


def _collection_name(config: RetrievalConfig) -> str:
    digest = hashlib.sha1(f"{config.chunk_key()}|{embeddings_id()}".encode()).hexdigest()[:10]
    return f"chart-{config.chunk_key()}-{digest}"


def build_vectorstore(config: RetrievalConfig = BASELINE, *, rebuild: bool = False, verbose: bool = True):
    """Return a vector store for `config`'s chunking. Built once, then reused from `.chroma/`. Thread-safe."""
    key = _collection_name(config)
    if key in _STORES and not rebuild:
        return _STORES[key]
    with _BUILD_LOCK:
        if key in _STORES and not rebuild:
            return _STORES[key]
        return _build_vectorstore(config, key, rebuild=rebuild, verbose=verbose)


def _build_vectorstore(config: RetrievalConfig, key: str, *, rebuild: bool, verbose: bool):
    embeddings = get_embeddings()
    chunks = chunk_corpus(config)
    ids = [f"{c.metadata['doc_id']}::{c.metadata['chunk']}" for c in chunks]
    store = None
    if os.getenv("LABS_VECTORSTORE", "chroma").lower() == "chroma":
        try:
            from langchain_chroma import Chroma

            store = Chroma(
                collection_name=key,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
                collection_metadata={"hnsw:space": "cosine"},
            )
            if rebuild:
                store.reset_collection()
            existing = store._collection.count()
            if existing != len(chunks):
                if existing:
                    store.reset_collection()
                for i in range(0, len(chunks), 128):
                    store.add_documents(chunks[i : i + 128], ids=ids[i : i + 128])
                if verbose:
                    print(f"Chroma: indexed {len(chunks)} chunks → .chroma/{key}")
            elif verbose:
                print(f"Chroma: reusing {existing} chunks from .chroma/{key}")
        except Exception as exc:  # noqa: BLE001: any Chroma problem → fall back, keep the lab running
            warnings.warn(f"Chroma unavailable ({type(exc).__name__}: {exc}); using InMemoryVectorStore.")
            store = None
    if store is None:
        from langchain_core.vectorstores import InMemoryVectorStore

        store = InMemoryVectorStore(embeddings)
        store.add_documents(chunks, ids=ids)
        if verbose:
            print(f"InMemoryVectorStore: indexed {len(chunks)} chunks")
    _STORES[key] = store
    return store


def _search(store, query: str, k: int, patient_id: str | None) -> list[tuple[Document, float]]:
    kwargs = {}
    if patient_id:
        if type(store).__name__ == "Chroma":
            kwargs["filter"] = {"patient_id": patient_id}
        else:
            kwargs["filter"] = lambda d: d.metadata.get("patient_id") == patient_id
    if type(store).__name__ == "Chroma":
        return store.similarity_search_with_relevance_scores(query, k=k, **kwargs)
    return store.similarity_search_with_score(query, k=k, **kwargs)  # InMemoryVectorStore: cosine similarity


# --------------------------------------------------------------------------------------------------
# Query rewriting (fix #3)
# --------------------------------------------------------------------------------------------------

ABBREVIATIONS = {
    "SOB": "shortness of breath", "DOE": "dyspnea on exertion", "c/o": "complains of", "CP": "chest pain",
    "NC": "nasal cannula", "RA": "room air", "HOB": "head of bed", "BLE": "bilateral lower extremities",
    "VSS": "vital signs stable", "AAOx3": "alert and oriented x3", "I&O": "intake and output",
    "UOP": "urine output", "CXR": "chest x-ray", "NPO": "nothing by mouth", "PRN": "as needed",
    "amb": "ambulated", "pt": "patient", "c": "with", "s": "without", "tol": "tolerated", "bil": "bilateral",
    "MD": "physician", "RN": "nurse", "add'l": "additional", "↑": "increased / raised", "↓": "decreased",
}

_REWRITE_PROMPT = """You rewrite a clinician's question into short search queries for a chart search engine.
Nursing notes use heavy shorthand, and later addenda can revise earlier findings. Produce 4 alternative
queries, one per line, no numbering:
1) the question in clinical shorthand (e.g. "shortness of breath" -> "c/o SOB", oxygen -> "NC", "RA")
2) the key clinical terms with common synonyms
3) a short keyword query
4) a query for later updates that could change the answer (addendum, result, removed, discontinued, revised)
Shorthand glossary: {glossary}

Question: {question}"""


def rewrite_query(question: str) -> list[str]:
    glossary = ", ".join(f"{k}={v}" for k, v in ABBREVIATIONS.items())
    try:
        text = get_model().invoke(_REWRITE_PROMPT.format(glossary=glossary, question=question)).content
        if isinstance(text, list):
            text = " ".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        variants = [re.sub(r"^\d+[).]\s*", "", ln.strip(" -•\t")) for ln in str(text).splitlines() if ln.strip()][:4]
    except Exception:  # noqa: BLE001: rewriting is an optimization; never fail retrieval because of it
        variants = []
    return [question, *variants]


# --------------------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------------------

def retrieve(question: str, patient_id: str | None, config: RetrievalConfig = BASELINE) -> list[Document]:
    """Top-k chunks for `question`, filtered to `patient_id` when given, with all of config's fixes."""
    store = build_vectorstore(config, verbose=False)
    if not config.query_rewrite:
        top = [doc for doc, _ in _search(store, question, config.k, patient_id)]
    else:
        # Reciprocal rank fusion: every query variant contributes its own top hits. Pooling raw similarity
        # scores instead would let plain-English documents crowd out shorthand-only matches (nursing notes).
        fused: dict[str, list] = {}
        for q in rewrite_query(question):
            for rank, (doc, _score) in enumerate(_search(store, q, config.k, patient_id)):
                key = f"{doc.metadata['doc_id']}::{doc.metadata['chunk']}"
                entry = fused.setdefault(key, [doc, 0.0])
                entry[1] += 1.0 / (60 + rank)
        ranked = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
        top = [doc for doc, _ in ranked[: config.k + 2]]
    if config.date_aware:
        top.sort(key=lambda d: d.metadata.get("date", ""), reverse=True)
    return top


def format_docs(docs: list[Document], config: RetrievalConfig = BASELINE) -> str:
    """Render retrieved chunks for the model. Each one gets a citation key: `[doc_id#section]`.
    The date-aware fix also shows each source's date, newest first."""
    if not docs:
        return "No matching passages found in this patient's record."
    blocks = []
    for d in docs:
        m = d.metadata
        cite = f"[source: {m['doc_id']}#{m['section']}]"
        if config.date_aware:
            cite += f" (date: {m['date']}, {m['doc_type']}, {m['department']})"
        blocks.append(f"{cite}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


DATE_AWARE_RULES = """
## Recency
Every source shows its date. Records change over time. When sources disagree, the MOST RECENT dated
document is the current truth; say what changed and when, and call out the older document as outdated.
For "current" questions, LEAD with the current state as of the newest document (e.g. "Currently on X,
started <date>"), then list anything discontinued or superseded separately, with its end date.
A note written "today" happened on that note's date. State dates for every fact you report."""


def config_summary(config: RetrievalConfig) -> dict:
    return asdict(config)
