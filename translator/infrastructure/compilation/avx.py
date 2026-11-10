"""Compilación con intrínsecos SIMD compatibles con la CPU anfitriona."""

from __future__ import annotations

from pathlib import Path

from translator.infrastructure.compilation.common import executable_path, resolve_c_compiler, run_compile


def avx_compile_argv(
    source_path: Path,
    extra_flags: list[str] | None = None,
    *,
    output_path: Path | None = None,
) -> list[str]:
    """Compiler command for an AVX intrinsics source file."""
    compiler = resolve_c_compiler(source_path)
    exe = output_path or executable_path(source_path)
    extra = extra_flags or []
    return [
        compiler,
        "-O3",
        "-march=native",
        "-fopenmp-simd",
        "-fopt-info-vec-optimized",
        "-Wall",
        *extra,
        str(source_path),
        "-o", str(exe),
        "-lm",
    ]


def compile_avx(
    source_path: Path,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Compile an AVX intrinsics source file.

    Flags:
    - ``-O3``: full optimisation (mandatory for SIMD throughput).
    - ``-march=native``: enable only the ISA extensions available on the host
      CPU. This selects the AVX-512, AVX2 or scalar preprocessor branch without
      emitting instructions unsupported by the machine that runs the binary.
    - ``-fopenmp-simd``: honour ``#pragma omp simd`` directives alongside
      explicit intrinsics, without requiring the full OpenMP runtime.
    - ``-fopt-info-vec-optimized``: log any additional loops the compiler
      auto-vectorises around the explicit intrinsic code.
    """
    return run_compile(avx_compile_argv(source_path, extra_flags), label="AVX")
