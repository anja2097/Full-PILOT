"""Pipeline domain entity — ordered sequence of translation and optimization stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from translator.domain.backend import Backend


# ---------------------------------------------------------------------------
# Accumulated paradigm profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParadigmProfile:
    """Accumulated parallelism paradigms active at a given pipeline stage.

    Computed by the registry from stage 0 up to the current stage (inclusive).
    Determines which compiler and runner to use regardless of the stage's
    logical purpose (translation vs. optimisation).

    Attributes:
        mpi:    At least one MPI stage has been applied. Compilation must use
                mpicc/mpicxx and execution must use mpirun.
        openmp: At least one OpenMP stage has been applied. Compilation must
                include -fopenmp.
        avx:    At least one AVX stage has been applied. Compilation must
                include -march=native.
    """

    mpi: bool = False
    openmp: bool = False
    avx: bool = False

    def with_mpi(self) -> "ParadigmProfile":
        return ParadigmProfile(mpi=True, openmp=self.openmp, avx=self.avx)

    def with_openmp(self) -> "ParadigmProfile":
        return ParadigmProfile(mpi=self.mpi, openmp=True, avx=self.avx)

    def with_avx(self) -> "ParadigmProfile":
        return ParadigmProfile(mpi=self.mpi, openmp=self.openmp, avx=True)

    def label(self) -> str:
        """Human-readable label, e.g. 'MPI+OpenMP+AVX'."""
        parts = []
        if self.mpi:
            parts.append("MPI")
        if self.openmp:
            parts.append("OpenMP")
        if self.avx:
            parts.append("AVX")
        return "+".join(parts) if parts else "serial"


# ---------------------------------------------------------------------------
# Stage prompts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StagePrompts:
    """Paths of the three prompt files for a single stage.

    Each path has already been resolved with pipeline-override fallback,
    so callers never need to implement fallback logic themselves.

    ``fix_execution`` is ``None`` when the prompt file does not exist, which
    disables ``--compare`` for that stage (same behaviour as the single-backend
    mode).
    """

    translate: Path
    fix_compile: Path
    fix_execution: Path | None


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineStage:
    """A single stage inside a pipeline, wrapping a Backend with resolved prompts.

    Attributes:
        backend:        Backend whose toolchain/prompts own this stage.
        prompts:        Resolved prompt file paths.
        stage_index:    0-based position in the pipeline.
        is_last:        True for the final stage.
        logical_name:   Token from the pipeline directory name, e.g. ``"OptMPI"``.
                        Distinct from ``backend.name`` (which is always ``"MPI"``).
        is_optimization: True for ``Opt*`` stages; they run via MoL instead of
                        the normal translate-and-fix loop.
        compile_variant: ``"serial"`` for ``OptSerial`` (compile without any
                        parallel flags, same as the serial baseline);
                        ``"translated"`` for all other stages.
        profile:        Accumulated paradigm profile *after* this stage is applied.
                        Used by the toolchain to select compiler and runner.
    """

    backend: "Backend"
    prompts: StagePrompts
    stage_index: int
    is_last: bool
    logical_name: str = ""            # e.g. "OptMPI", "MPI", "AVX"
    is_optimization: bool = False
    compile_variant: str = "translated"   # "serial" | "translated"
    profile: ParadigmProfile = field(default_factory=ParadigmProfile)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pipeline:
    """Ordered sequence of PipelineStages, discovered dynamically from prompts/pipelines/."""

    name: str                                  # e.g. "AVX-OpenMP-MPI"
    stages: tuple[PipelineStage, ...]

    @property
    def final_backend(self) -> "Backend":
        """The Backend of the last stage — used for execution and benchmarking."""
        return self.stages[-1].backend

    @property
    def final_profile(self) -> ParadigmProfile:
        """The accumulated paradigm profile after all stages."""
        return self.stages[-1].profile

    def accumulated_profile(self, up_to_stage: int) -> ParadigmProfile:
        """Return the accumulated paradigm profile up to and including *up_to_stage*."""
        return self.stages[up_to_stage].profile

    def artifact_path(self, source_path: Path, up_to_stage: int) -> Path:
        """Return the output path for the stage at *up_to_stage* (0-based).

        The name is built from the ``logical_name`` tokens of all stages up to
        and including *up_to_stage*, lower-cased and joined by ``_``.

        Examples for ``OptSerial-MPI-OptMPI-OpenMP-OptOpenMP-AVX-OptAVX``::

            stage 0 (OptSerial):  code_optserial.c
            stage 1 (MPI):        code_optserial_mpi.c
            stage 2 (OptMPI):     code_optserial_mpi_optmpi.c
            …

        For legacy pipelines without ``Opt`` stages the behaviour is unchanged::

            stage 0 (AVX):        code_avx.c
            stage 1 (OpenMP):     code_avx_openmp.c
            stage 2 (MPI):        code_avx_openmp_mpi.c
        """
        tokens = [s.logical_name.lower() for s in self.stages[: up_to_stage + 1]]
        last_backend = self.stages[up_to_stage].backend
        ext = last_backend.extension or source_path.suffix
        suffix = "_".join(tokens)
        return source_path.with_name(f"{source_path.stem}_{suffix}{ext}")
