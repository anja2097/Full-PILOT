"""Compilation façade: dispatches to the appropriate backend toolchain."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from translator.domain.backend import Backend
from translator.infrastructure.compilation import avx, ghighway, go, kokkos, mpi, openmp, vectorize
from translator.infrastructure.compilation.common import executable_path, resolve_c_compiler, run_compile
from translator.infrastructure.compilation.simd_syntax import check_simd_branches

if TYPE_CHECKING:
    from translator.domain.pipeline import ParadigmProfile


def compile_serial(
    source_path: Path,
    *,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Compile the serial source file with true serial flags (GCC/G++ -O3, no parallelism).

    This is the correct serial baseline: no -fopenmp, no -march=native, no MPI.
    Used for the initial serial timing reference and for OptSerial compilation.
    """
    compiler = resolve_c_compiler(source_path)
    exe = executable_path(source_path)
    extra = extra_flags or []
    return run_compile(
        [compiler, "-O3", "-Wall", *extra, str(source_path), "-o", str(exe), "-lm"],
        label="serial",
    )


def translation_compile_argv(
    source_path: Path,
    backend: Backend,
    *,
    variant: str = "translated",
    extra_flags: list[str] | None = None,
    profile: "ParadigmProfile | None" = None,
    hybrid_openmp: bool = False,
    hybrid_avx: bool = False,
    output_path: Path | None = None,
) -> tuple[list[str], str] | None:
    """Compiler command for C/C++ stages, or None for Go, Kokkos and Highway.

    ``variant="serial"`` ignores *profile* and uses plain ``-O3``.
    """
    if variant == "serial":
        compiler = resolve_c_compiler(source_path)
        exe = output_path or executable_path(source_path)
        extra = extra_flags or []
        return (
            [compiler, "-O3", "-Wall", *extra, str(source_path), "-o", str(exe), "-lm"],
            "serial",
        )

    extra = extra_flags or []
    if profile is not None and profile.mpi:
        return mpi.mpi_compile_argv(
            source_path, extra,
            hybrid_openmp=profile.openmp,
            hybrid_avx=profile.avx,
            output_path=output_path,
        )

    eff_openmp = profile.openmp if profile is not None else hybrid_openmp
    eff_avx = profile.avx if profile is not None else hybrid_avx
    slug = backend.slug
    if slug == "openmp":
        return openmp.openmp_compile_argv(source_path, extra, output_path=output_path), "OpenMP"
    if slug == "avx":
        return avx.avx_compile_argv(source_path, extra, output_path=output_path), "AVX"
    if slug == "mpi":
        return mpi.mpi_compile_argv(
            source_path, extra,
            hybrid_openmp=eff_openmp, hybrid_avx=eff_avx, output_path=output_path,
        )
    if slug == "vectorize":
        return (
            vectorize.vectorize_compile_argv(source_path, extra, output_path=output_path),
            "Vectorize",
        )
    return None


def compile_file(
    source_path: Path,
    backend: Backend,
    *,
    variant: str = "translated",
    extra_flags: list[str] | None = None,
    profile: "ParadigmProfile | None" = None,
    # Legacy parameters kept for single-backend mode and backward compatibility.
    # When *profile* is provided they are ignored.
    hybrid_openmp: bool = False,
    hybrid_avx: bool = False,
) -> tuple[bool, str]:
    """Compile a source file for the given backend.

    Args:
        variant:      ``"serial"`` compiles the original with serial flags
                      (no OpenMP, no MPI, GCC/G++ -O3);
                      ``"translated"`` uses the backend-specific toolchain.
        profile:      Accumulated paradigm profile from the pipeline.  When
                      provided it takes precedence over ``hybrid_openmp`` and
                      ``hybrid_avx``.  If ``profile.mpi`` is True the MPI
                      toolchain (mpicc/mpicxx) is used regardless of
                      ``backend.slug``, with ``-fopenmp`` if ``profile.openmp``
                      and ``-march=native`` if ``profile.avx``.
        hybrid_openmp: Legacy flag — compile MPI stages with ``-fopenmp``.
                       Ignored when *profile* is provided.
        hybrid_avx:    Legacy flag — compile MPI stages with AVX-512/AVX2 flags.
                       Ignored when *profile* is provided.
    """
    built = translation_compile_argv(
        source_path, backend,
        variant=variant,
        extra_flags=extra_flags,
        profile=profile,
        hybrid_openmp=hybrid_openmp,
        hybrid_avx=hybrid_avx,
    )
    if built is not None:
        cmd, label = built
        ok, err = run_compile(cmd, label=label)
        if not ok:
            return False, err
        if variant != "serial" and profile is not None and profile.avx:
            syn_ok, syn_err = check_simd_branches(source_path, cmd)
            if not syn_ok:
                return False, syn_err
        return True, ""

    extra = extra_flags or []
    dispatch: dict[str, Callable[[], tuple[bool, str]]] = {
        "kokkos":    lambda: kokkos.compile_kokkos(source_path, extra_flags=extra),
        "go":        lambda: go.compile_go(source_path),
        "ghighway":  lambda: ghighway.compile_ghighway(source_path, extra_flags=extra),
    }
    compile_fn = dispatch.get(backend.slug)
    if compile_fn is None:
        raise ValueError(f"Backend has no compilation toolchain: {backend.name!r}")
    return compile_fn()


__all__ = ["compile_file", "compile_serial", "executable_path", "translation_compile_argv"]
