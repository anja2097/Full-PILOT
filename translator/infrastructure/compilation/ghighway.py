"""Compilación con Google Highway."""

from __future__ import annotations

from pathlib import Path

from translator.infrastructure.compilation.common import (
    executable_path,
    resolve_c_compiler,
    run_compile,
)


def compile_ghighway(
    source_path: Path,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    compiler = resolve_c_compiler(source_path, cxx=True)
    exe = executable_path(source_path)
    extra = extra_flags or []
    cmd = [
        compiler,
        "-O3",
        "-std=c++17",
        *extra,
        str(source_path),
        "-o",
        str(exe),
        "-lhwy",
        "-lm",
    ]
    return run_compile(cmd, label="GHighway")
