"""Static analysis test: verify zero undefined names across scripts/ and src/rotapetro/."""
from __future__ import annotations

import ast
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_no_undefined_names_in_scripts_and_src():
    """Run pyflakes across scripts/ and src/rotapetro/ and fail on any undefined name."""
    pyflakes = pytest.importorskip("pyflakes")
    from pyflakes import checker, messages

    errors: list[str] = []
    folders = [_PROJECT_ROOT / "scripts", _PROJECT_ROOT / "src" / "rotapetro"]

    for folder in folders:
        for py_path in sorted(folder.rglob("*.py")):
            try:
                code = py_path.read_text(encoding="utf-8")
                tree = ast.parse(code, filename=str(py_path))
                chk = checker.Checker(tree, filename=str(py_path))
                for msg in chk.messages:
                    if isinstance(msg, messages.UndefinedName) or "undefined name" in str(msg).lower():
                        errors.append(f"{py_path.relative_to(_PROJECT_ROOT)}:{msg.lineno}: {msg}")
            except Exception as exc:
                errors.append(f"{py_path.relative_to(_PROJECT_ROOT)}: parse error {exc}")

    assert not errors, f"Found {len(errors)} undefined name(s):\n" + "\n".join(errors)
