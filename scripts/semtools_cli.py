#!/usr/bin/env python3
"""
Semtools CLI - ingest files under a work_dir with TEI embeddings into pgvector.

Usage:
    scripts/semtools_cli.py --work-dir frontend/ui --glob "**/*.md" --collection knowledge

Env:
    ZEN_PG_DSN=postgresql://postgres:postgres@localhost:5432/zen
    TEI_URL=http://localhost:8082
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from pathlib import Path as _Path

# Load semtools module directly from file to avoid importing the full
# 'tools' package (which pulls in unrelated modules and async code).
_SEMTOOLS_PATH = _Path(__file__).resolve().parents[1] / "tools" / "semtools.py"
_spec = importlib.util.spec_from_file_location("_semtools_isolated", str(_SEMTOOLS_PATH))
if _spec is None or _spec.loader is None:
    raise SystemExit(f"Failed to load semtools from {_SEMTOOLS_PATH}")
_semtools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_semtools)
ingest_documents = _semtools.ingest_documents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, help="Repo-relative work dir")
    ap.add_argument("--glob", default="**/*", help="Glob pattern relative to work_dir")
    ap.add_argument("--collection", default="code", help="Collection: code, knowledge, other")
    ap.add_argument("--tei-url", default=os.getenv("TEI_URL", "http://localhost:8082"))
    ap.add_argument("--provider", default=os.getenv("EMBEDDINGS_PROVIDER", "tei"), choices=["tei","ollama","openrouter"], help="Embeddings provider to use")
    ap.add_argument("--model", default=os.getenv("OLLAMA_EMBED_MODEL", os.getenv("EMBEDDINGS_MODEL_OPENROUTER", "")), help="Embedding model (provider-specific)")
    ap.add_argument("--dim", type=int, default=int(os.getenv("RAG_VECTOR_DIM", "384")), help="Vector dimension for pgvector")
    args = ap.parse_args()

    repo_root = Path.cwd()
    wdir = (repo_root / args.work_dir).resolve()
    if not wdir.is_dir():
        print(f"work_dir not found: {wdir}", file=sys.stderr)
        sys.exit(2)

    docs = []
    for p in wdir.glob(args.glob):
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            doc_id = str(p.relative_to(repo_root))
            meta = {"path": str(p.relative_to(repo_root))}
            docs.append((doc_id, text, meta))

    if not docs:
        print("No files matched.")
        return 0

    # Propagate overrides via environment (semtools reads env)
    os.environ["EMBEDDINGS_PROVIDER"] = args.provider
    if args.provider == "ollama" and args.model:
        os.environ.setdefault("OLLAMA_EMBED_MODEL", args.model)
    if args.provider == "openrouter" and args.model:
        os.environ.setdefault("EMBEDDINGS_MODEL_OPENROUTER", args.model)
    os.environ["RAG_VECTOR_DIM"] = str(args.dim)

    scope = {"work_dir": args.work_dir}
    ingest_documents(docs, scope, collection=args.collection, tei_url=args.tei_url)
    print(f"Ingested {len(docs)} docs into collection {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
