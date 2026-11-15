import os
import sys

# Ensure backend directory is on sys.path for Vercel Serverless runtime
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.join(_ROOT, "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from main import app  # noqa: E402, F401
