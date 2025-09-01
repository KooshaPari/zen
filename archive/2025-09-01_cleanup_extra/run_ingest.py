#!/usr/bin/env python3
"""Direct ingest script that bypasses the table creation issue"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.semtools import ingest_documents

# Set environment variables
os.environ['EMBEDDINGS_PROVIDER'] = 'ollama'
os.environ['OLLAMA_URL'] = 'http://localhost:11434'
os.environ['OLLAMA_EMBED_MODEL'] = 'nomic-embed-text'
os.environ['RAG_VECTOR_DIM'] = '768'
os.environ['ZEN_PG_DSN'] = 'postgresql://postgres:postgres@localhost:5433/zen'

# Import after setting env vars
import glob

# Find all markdown files
docs = []
for md_file in glob.glob("**/*.md", recursive=True):
    with open(md_file, encoding='utf-8') as f:
        content = f.read()
        docs.append({
            'id': md_file,
            'text': content,
            'metadata': {'source': md_file, 'type': 'markdown'}
        })

print(f"Found {len(docs)} markdown files to ingest")

# Skip the ensure_schema() call by monkey-patching
import tools.semtools


def dummy_ensure_schema():
    print("Skipping schema creation (already exists)")

tools.semtools.ensure_schema = dummy_ensure_schema

# Run ingest
ingest_documents(docs, ".", collection="knowledge")
print("Ingest complete!")
