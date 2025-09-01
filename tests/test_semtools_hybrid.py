import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OS_TESTS") != "1",
    reason="Set RUN_OS_TESTS=1 to run OpenSearch hybrid tests"
)


def test_rrf_fuse_smoke():
    from tools.semtools_bm25 import rrf_fuse
    dense = [
        {"id": "1", "text": "a", "metadata": {}},
        {"id": "2", "text": "b", "metadata": {}},
    ]
    sparse = [
        {"id": "3", "text": "c", "metadata": {}},
        {"id": "1", "text": "a", "metadata": {}},
    ]
    out = rrf_fuse(dense, sparse, k=60, top_k=3)
    assert isinstance(out, list)
    assert out

