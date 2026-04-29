from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

project = "SPatho"
author = "Taobo Hu"
copyright = "2026, Taobo Hu"


def _read_version() -> str:
    init_file = ROOT / "src" / "spatho" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init_file.read_text(encoding="utf-8"))
    return match.group(1) if match else "0.0.0"


release = _read_version()
version = release

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "SPatho Documentation"
html_static_path = ["_static"]

myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]
