"""Compilación con MPI."""

from __future__ import annotations

import shutil
from pathlib import Path

from translator.exceptions import CompilerNotFoundError
from translator.infrastructure.compilation.common import executable_path, run_compile


def _mpi_compiler(source_path: Path) -> str:
    if source_path.suffix == ".cpp":
        candidates = ["mpicxx", "mpiCC", "mpic++"]
    else:
        candidates = ["mpicc"]
    for c in candidates:
        if shutil.which(c):
            return c
    raise CompilerNotFoundError("No MPI compiler (mpicc/mpicxx) found in PATH.")


def mpi_compile_argv(
    source_path: Path,
    extra_flags: list[str] | None = None,
    *,
    hybrid_openmp: bool = False,
    hybrid_avx: bool = False,
    output_path: Path | None = None,
) -> tuple[list[str], str]:
    """Return ``(command, label)`` for an MPI compile."""
    compiler = _mpi_compiler(source_path)
    exe = output_path or executable_path(source_path)
    extra = extra_flags or []
    flags = ["-O3", "-Wall"]
    if hybrid_openmp:
        flags.append("-fopenmp")
    if hybrid_avx:
        flags.append("-march=native")
    label_parts = ["MPI"]
    if hybrid_openmp:
        label_parts.append("OpenMP")
    if hybrid_avx:
        label_parts.append("AVX")
    cmd = [compiler, *flags, *extra, str(source_path), "-o", str(exe), "-lm"]
    return cmd, "+".join(label_parts)


def compile_mpi(
    source_path: Path,
    extra_flags: list[str] | None = None,
    *,
    hybrid_openmp: bool = False,
    hybrid_avx: bool = False,
) -> tuple[bool, str]:
    """Compile an MPI source file.

    Args:
        hybrid_openmp: When ``True``, add ``-fopenmp`` to enable hybrid
            MPI+OpenMP mode (one MPI process per node, OpenMP threads within).
        hybrid_avx: When ``True``, use ``-march=native`` so explicit SIMD
            intrinsics introduced by a prior AVX stage are compiled only for
            the ISA extensions supported by the host CPU.
    """
    cmd, label = mpi_compile_argv(
        source_path, extra_flags,
        hybrid_openmp=hybrid_openmp, hybrid_avx=hybrid_avx,
    )
    return run_compile(cmd, label=label)
