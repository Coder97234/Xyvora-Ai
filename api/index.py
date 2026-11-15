import io
import os
import sys
import tarfile
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_BACKEND = os.path.join(_ROOT, "backend")

if os.path.isfile(os.path.join(_LOCAL_BACKEND, "main.py")):
    _BACKEND_PATH = _LOCAL_BACKEND
else:
    _TMP_REPO = "/tmp/xyvora_ai_src"
    _BACKEND_PATH = os.path.join(_TMP_REPO, "backend")
    if not os.path.isfile(os.path.join(_BACKEND_PATH, "main.py")):
        ref = os.environ.get("XYVORA_GIT_REF", "arena/193fdff4-xyvora-ai")
        url = f"https://codeload.github.com/Coder97234/Xyvora-Ai/tar.gz/{ref}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read()
        os.makedirs(_TMP_REPO, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            for member in tf.getmembers():
                parts = member.name.split("/", 1)
                if len(parts) == 2 and parts[1]:
                    member.name = parts[1]
                    tf.extract(member, _TMP_REPO)

if _BACKEND_PATH not in sys.path:
    sys.path.insert(0, _BACKEND_PATH)

from main import app  # noqa: E402, F401
