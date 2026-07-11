"""Test env setup — must run before core.config is imported anywhere."""

import os
import tempfile
from pathlib import Path

# Never download models in unit tests; keep test index data out of the repo.
os.environ.setdefault("EAGER_LOAD_MODELS", "false")
# Satisfies the startup fail-fast on weak JWT secrets without needing a .env.
os.environ.setdefault("JWT_SECRET", "unit-test-secret-0123456789abcdef")
_tmp = Path(tempfile.mkdtemp(prefix="opkey-test-index-"))
os.environ.setdefault("CHROMA_DIR", str(_tmp / "chroma"))
os.environ.setdefault("BM25_PATH", str(_tmp / "chroma" / "bm25.pkl"))
