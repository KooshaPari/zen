#!/usr/bin/env python3
"""
One-shot migration for embedding vector dimensions.

Use when switching embedding models (e.g., 384→768 for nomic-embed-text).

Features:
- Migrates rag_chunks.embedding by re-embedding text using the configured provider
  (EMBEDDINGS_PROVIDER=ollama|openrouter|tei)
- Optionally resets zen_shared.* embedding columns to the new dimension
  (vectors.embedding, conversation_memory.content_embedding, model_performance.task_embedding)
  and re-embeds vectors.content when --recompute-zen-shared is set

Environment:
  ZEN_PG_DSN           postgresql://user:pass@host:port/db (default: postgresql://postgres:postgres@localhost:5432/zen)
  RAG_VECTOR_DIM       target dimension (e.g., 768)
  EMBEDDINGS_PROVIDER  tei|ollama|openrouter
  TEI_URL              when provider=tei
  OLLAMA_URL           when provider=ollama (default http://localhost:11434)
  OLLAMA_EMBED_MODEL   model for ollama (default nomic-embed-text)
  OPENROUTER_API_KEY   when provider=openrouter
  EMBEDDINGS_MODEL_OPENROUTER  model for openrouter (default text-embedding-3-small)

Usage:
  python scripts/migrate_embeddings_dim.py --only rag --batch-size 100
  python scripts/migrate_embeddings_dim.py --all --recompute-zen-shared --batch-size 50
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover
    print("psycopg is required: pip install psycopg[binary]", file=sys.stderr)
    raise


def _dsn() -> str:
    return os.getenv("ZEN_PG_DSN", "postgresql://postgres:postgres@localhost:5432/zen")


def _target_dim() -> int:
    try:
        return int(os.getenv("RAG_VECTOR_DIM", "768"))
    except Exception:
        return 768


def _vec_text(vec: list[float]) -> str:
    # pgvector textual format: '[x,y,z]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _embed_batch(texts: list[str]) -> list[list[float]]:
    # Reuse semtools provider plumbing
    from tools.semtools import embed_texts
    return embed_texts(texts)


def migrate_rag_chunks(conn: psycopg.Connection, batch_size: int = 100) -> None:
    dim = _target_dim()
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE IF EXISTS rag_chunks ADD COLUMN IF NOT EXISTS embedding_new vector({dim})")
        conn.commit()

    total = 0
    while True:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, text
                FROM rag_chunks
                WHERE embedding_new IS NULL
                ORDER BY id
                LIMIT %s
                """,
                (batch_size,),
            )
            rows: list[dict] = cur.fetchall()
        if not rows:
            break
        texts = [r["text"] for r in rows]
        ids = [r["id"] for r in rows]
        embs = _embed_batch(texts)

        # Update in a single transaction
        with conn.cursor() as cur:
            for i, emb in enumerate(embs):
                cur.execute(
                    "UPDATE rag_chunks SET embedding_new = %s::vector WHERE id = %s",
                    (_vec_text(emb), ids[i]),
                )
        conn.commit()
        total += len(rows)
        print(f"rag_chunks: migrated {total} rows…")

    # Swap columns
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='rag_chunks' AND column_name='embedding'")
        if cur.fetchone():
            cur.execute("ALTER TABLE rag_chunks RENAME COLUMN embedding TO embedding_old")
        cur.execute("ALTER TABLE rag_chunks RENAME COLUMN embedding_new TO embedding")
        conn.commit()
    print("rag_chunks: migration complete")


def _exists_table(cur: psycopg.Cursor, schema: str, table: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s",
        (schema, table),
    )
    return cur.fetchone() is not None


def reset_zen_shared(conn: psycopg.Connection, recompute: bool, batch_size: int = 100) -> None:
    dim = _target_dim()
    with conn.cursor() as cur:
        # vectors
        if _exists_table(cur, "zen_shared", "vectors"):
            print("zen_shared.vectors: preparing new column…")
            cur.execute(f"ALTER TABLE zen_shared.vectors ADD COLUMN IF NOT EXISTS embedding_new vector({dim})")
        # conversation_memory
        if _exists_table(cur, "zen_shared", "conversation_memory"):
            print("zen_shared.conversation_memory: resetting content_embedding…")
            cur.execute("ALTER TABLE zen_shared.conversation_memory DROP COLUMN IF EXISTS content_embedding")
            cur.execute(f"ALTER TABLE zen_shared.conversation_memory ADD COLUMN content_embedding vector({dim})")
        # model_performance
        if _exists_table(cur, "zen_shared", "model_performance"):
            print("zen_shared.model_performance: resetting task_embedding…")
            cur.execute("ALTER TABLE zen_shared.model_performance DROP COLUMN IF EXISTS task_embedding")
            cur.execute(f"ALTER TABLE zen_shared.model_performance ADD COLUMN task_embedding vector({dim})")
        conn.commit()

    if recompute and _exists_table(conn.cursor(), "zen_shared", "vectors"):
        # Re-embed vectors.content into embedding_new
        total = 0
        while True:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT vector_id, content
                    FROM zen_shared.vectors
                    WHERE embedding_new IS NULL
                    ORDER BY vector_id
                    LIMIT %s
                    """,
                    (batch_size,),
                )
                rows = cur.fetchall()
            if not rows:
                break
            texts = [r["content"] for r in rows]
            ids = [r["vector_id"] for r in rows]
            embs = _embed_batch(texts)
            with conn.cursor() as cur:
                for i, emb in enumerate(embs):
                    cur.execute(
                        "UPDATE zen_shared.vectors SET embedding_new = %s::vector WHERE vector_id = %s",
                        (_vec_text(emb), ids[i]),
                    )
            conn.commit()
            total += len(rows)
            print(f"zen_shared.vectors: migrated {total} rows…")

        # Swap columns
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.columns WHERE table_schema='zen_shared' AND table_name='vectors' AND column_name='embedding'")
            if cur.fetchone():
                cur.execute("ALTER TABLE zen_shared.vectors RENAME COLUMN embedding TO embedding_old")
            cur.execute("ALTER TABLE zen_shared.vectors RENAME COLUMN embedding_new TO embedding")
            conn.commit()
        print("zen_shared.vectors: migration complete")


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate embedding dimensions by re-embedding content")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--only", choices=["rag", "zen"], help="Migrate only rag_chunks or only zen_shared*")
    g.add_argument("--all", action="store_true", help="Migrate both rag_chunks and zen_shared*")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--recompute-zen-shared", action="store_true", help="Recompute zen_shared.vectors embeddings from content")
    args = ap.parse_args()

    print(f"Target dimension: {_target_dim()}")
    print(f"Provider: {os.getenv('EMBEDDINGS_PROVIDER', 'tei')}\n")

    with psycopg.connect(_dsn()) as conn:
        if args.only == "rag":
            migrate_rag_chunks(conn, args.batch_size)
        elif args.only == "zen":
            reset_zen_shared(conn, args.recompute_zen_shared, args.batch_size)
        else:
            migrate_rag_chunks(conn, args.batch_size)
            reset_zen_shared(conn, args.recompute_zen_shared, args.batch_size)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

