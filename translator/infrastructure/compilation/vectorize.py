"""Compilación con autovectorización GCC."""

from __future__ import annotations

from pathlib import Path

from translator.infrastructure.compilation.common import executable_path, resolve_c_compiler, run_compile


def vectorize_compile_argv(
    source_path: Path,
    extra_flags: list[str] | None = None,
    *,
    output_path: Path | None = None,
) -> list[str]:
    compiler = resolve_c_compiler(source_path)
    exe = output_path or executable_path(source_path)
    extra = extra_flags or []
    return [
        compiler,
        "-O3",
        "-ftree-vectorize",
        "-march=native",
        "-fopenmp-simd",
        "-fopt-info-vec-optimized",
        "-Wall",
        *extra,
        str(source_path),
        "-o", str(exe),
        "-lm",
    ]


def compile_vectorize(
    source_path: Path,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Compile with GCC auto-vectorisation flags and SIMD pragma support.

    Flags:
    - ``-O3``: base optimisation level.
    - ``-ftree-vectorize``: enable tree-based auto-vectorisation (redundant at O2 but explicit).
    - ``-march=native``: generate code tuned for the host CPU's vector ISA.
    - ``-fopenmp-simd``: honour ``#pragma omp simd`` directives without full OpenMP runtime.
    - ``-fopt-info-vec-optimized``: emit a report of successfully vectorised loops.
    """
    return run_compile(vectorize_compile_argv(source_path, extra_flags), label="Vectorize")
