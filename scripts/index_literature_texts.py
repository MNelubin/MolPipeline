"""Index plain-text literature files into the MolPipeline RAG store.

The default mode is keyword/BM25-ready indexing: it writes text chunks into
Chroma and parent chunks into SQLite without requiring an embedding model.
Use --with-embeddings only when sentence-transformers is available and vector
search is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp.rag.embeddings import LiteratureEmbedder
from mvp.rag.models import DocumentSource, SectionType
from mvp.rag.tracking import TrackingDB

COLLECTION_NAME = "literature_chunks"


def _clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunks(words: list[str], size: int, overlap: int) -> list[str]:
    if not words:
        return []
    out: list[str] = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + size]).strip()
        if chunk:
            out.append(chunk)
        if start + size >= len(words):
            break
    return out


def _doc_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"manual:{path.stem}:{digest}"


def _title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def index_text_files(
    source_dir: Path,
    vectordb_dir: Path,
    tracking_db_path: Path,
    *,
    force: bool = False,
    with_embeddings: bool = False,
) -> int:
    import chromadb

    text_files = sorted(source_dir.rglob("*.txt"))
    if not text_files:
        print(f"No .txt files found in {source_dir}")
        return 0

    client = chromadb.PersistentClient(path=str(vectordb_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    tracking = TrackingDB(tracking_db_path)
    embedder = LiteratureEmbedder() if with_embeddings else None
    indexed_count = 0

    for path in text_files:
        doc_id = _doc_id(path)
        title = _title(path)
        if tracking.is_indexed(doc_id) and not force:
            print(f"skip indexed: {path}")
            continue

        if force:
            try:
                collection.delete(where={"doc_id": doc_id})
            except Exception:
                pass

        text = _clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        parent_texts = _chunks(text.split(), size=900, overlap=120)
        child_ids: list[str] = []
        child_texts: list[str] = []
        child_metas: list[dict[str, str | int | float | bool]] = []
        parent_rows: list[tuple[str, str, str, str, str]] = []

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = f"{doc_id}:p{parent_index}"
            parent_rows.append((
                parent_id,
                doc_id,
                SectionType.OTHER.value,
                parent_text,
                json.dumps({"title": title, "path": str(path)}, ensure_ascii=False),
            ))
            for child_index, child_text in enumerate(_chunks(parent_text.split(), size=260, overlap=60)):
                child_ids.append(f"{parent_id}:c{child_index}")
                child_texts.append(child_text)
                child_metas.append({
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "title": title,
                    "source": DocumentSource.MANUAL.value,
                    "section": SectionType.OTHER.value,
                    "path": str(path),
                })

        if not child_ids:
            tracking.mark_failed(doc_id, DocumentSource.MANUAL)
            print(f"empty: {path}")
            continue

        if embedder:
            embeddings = embedder.embed_texts(child_texts).tolist()
        else:
            embeddings = [[0.0] for _ in child_texts]

        collection.upsert(
            ids=child_ids,
            documents=child_texts,
            metadatas=child_metas,
            embeddings=embeddings,
        )
        tracking.store_parent_chunks(parent_rows)
        tracking.mark_indexed(doc_id, DocumentSource.MANUAL, title=title, chunk_count=len(child_ids))
        indexed_count += 1
        print(f"indexed: {path} ({len(child_ids)} chunks)")

    tracking.close()
    print(f"Indexed {indexed_count} document(s)")
    return indexed_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("validation_materials/sources"))
    parser.add_argument("--vectordb-dir", type=Path, default=Path("data/vectordb/literature"))
    parser.add_argument("--tracking-db", type=Path, default=Path("data/literature_tracking.db"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--with-embeddings", action="store_true")
    args = parser.parse_args()

    index_text_files(
        args.source_dir,
        args.vectordb_dir,
        args.tracking_db,
        force=args.force,
        with_embeddings=args.with_embeddings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
