"""Compilación OpenMP y serial genérica."""

from __future__ import annotations

from pathlib import Path

from translator.infrastructure.compilation.common import (
    executable_path,
    resolve_c_compiler,
    run_compile,
)


def openmp_compile_argv(
    source_path: Path,
    extra_flags: list[str] | None = None,
    *,
    output_path: Path | None = None,
) -> list[str]:
    compiler = resolve_c_compiler(source_path)
    exe = output_path or executable_path(source_path)
    extra = extra_flags or []
    return [
        compiler, "-O3", "-Wall", "-fopenmp", "-march=native",
        *extra, str(source_path), "-o", str(exe), "-lm",
    ]


def compile_openmp(
    source_path: Path,
    *,
    serial: bool,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    label = "serial" if serial else "OpenMP"
    return run_compile(openmp_compile_argv(source_path, extra_flags), label=label)
