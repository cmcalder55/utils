# Auto-add project base path to sys.path

import sys
from pathlib import Path

def find_project_root(marker_files=("requirements.txt", ".git", "pyproject.toml")) -> Path:
    """Traverse parent directories to find project root based on known marker files or folders."""
    current = Path(__file__ if "__file__" in globals() else ".").resolve()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise RuntimeError("Project root not found. Please place a marker file like requirements.txt or .git in the root.")

try:
    PROJECT_ROOT = find_project_root().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print(f"✔ Project base path added: {PROJECT_ROOT}")
except RuntimeError as e:
    print(f"⚠ {e}")
