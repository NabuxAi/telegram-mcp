import os
import sys
import tempfile
from pathlib import Path

# The telegram_notebook package lives at the repository root in this merged project.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Point any accidental settings load at a throwaway dir, never the repo.
_tmp = tempfile.mkdtemp(prefix="tgnb-test-")
os.environ.setdefault("DATA_DIR", _tmp)
os.environ.setdefault("DB_PATH", os.path.join(_tmp, "store.db"))
os.environ.setdefault("MEDIA_DIR", os.path.join(_tmp, "media"))
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
