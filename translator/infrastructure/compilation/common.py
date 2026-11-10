"""Shared utilities for C/C++ compilation."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from translator.exceptions import CompilerNotFoundError

logger = logging.getLogger(__name__)


def resolve_c_compiler(source_path: Path, *, cxx: bool = False) -> str:
    if cxx or source_path.suffix == ".cpp":
        candidates = ["g++", "g++-15"]
    else:
        candidates = ["gcc", "gcc-15"]
    for c in candidates:
        if shutil.which(c):
            return c
    raise CompilerNotFoundError("No C/C++ compiler found in PATH.")


def run_compile(cmd: list[str], *, label: str) -> tuple[bool, str]:
    logger.info("\n\n  Compiling (%s): %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        logger.info("  Result: compilation OK")
        return True, ""
    logger.info("  Result: compilation error")
    return False, output


def executable_path(source_path: Path) -> Path:
    return source_path.with_suffix("")
