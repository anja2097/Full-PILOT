"""Compilación con Go."""

from __future__ import annotations

import shutil
from pathlib import Path

from translator.exceptions import CompilerNotFoundError
from translator.infrastructure.compilation.common import executable_path, run_compile


def compile_go(source_path: Path) -> tuple[bool, str]:
    if not shutil.which("go"):
        raise CompilerNotFoundError("'go' was not found in PATH.")
    exe = executable_path(source_path)
    cmd = ["go", "build", "-o", str(exe), str(source_path)]
    return run_compile(cmd, label="Go")
