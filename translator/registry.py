"""Backend registry — discovers available backends and resolves them by name."""

from __future__ import annotations

import logging
from pathlib import Path

from translator.config.env import get_env
from translator.config.settings import PROMPTS_BACKENDS_DIR, PROMPTS_PIPELINES_DIR, PROMPTS_DIR
from translator.domain.backend import Backend
from translator.domain.pipeline import (
    ParadigmProfile,
    Pipeline,
    PipelineStage,
    StagePrompts,
)

logger = logging.getLogger(__name__)

# Opt* stages without their own fix_execution prompt borrow the sibling
# translation stage in the same pipeline, not prompts/backends/.
_OPT_FIX_SIBLING = {
    "optavx": "AVX",
    "optopenmp": "OpenMP",
    "optmpi": "MPI",
}

# Extension produced by each backend slug (default: inherit from source file).
# Slugs absent from this map use the source extension unchanged.
_EXTENSION_BY_SLUG: dict[str, str] = {
    "kokkos": ".cpp",
    "ghighway": ".cpp",
    "go": ".go",
}

# Extra environment variables injected at runtime for specific backends.
_RUN_ENV_BY_SLUG: dict[str, dict[str, str]] = {
    "kokkos": {
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
    },
}

# Which paradigm flag a translation-stage slug adds to the accumulated profile.
# Slugs absent from this map do not add any paradigm flag.
_SLUG_TO_PARADIGM: dict[str, str] = {
    "mpi": "mpi",
    "openmp": "openmp",
    "avx": "avx",
    "vectorize": "avx",   # Vectorize produces AVX-like SIMD code
    "ghighway": "avx",    # Google Highway — SIMD
}

# For Opt* stages: which paradigm slug the Opt must immediately follow.
# E.g. "OptMPI" must follow a stage whose paradigm is "mpi".
_OPT_SUFFIX_TO_PARADIGM: dict[str, str] = {
    "optmpi": "mpi",
    "optopenmp": "openmp",
    "optavx": "avx",
    # "optserial" has no prior-paradigm constraint (must be first stage)
}


def _build_backend(subdir: Path) -> Backend:
    """Construct a Backend from a prompts sub-directory, injecting runtime config."""
    name = subdir.name
    slug = name.lower()
    extension = _EXTENSION_BY_SLUG.get(slug, "")  # "" means inherit source extension

    run_prefix: list[str] = []
    if slug == "mpi":
        nprocs = get_env("MPI_PROCS", "4")
        run_prefix = ["mpirun", "-bind-to", "core:4", "-genv", "OMP_NUM_THREADS", "4", "-np", nprocs]

    return Backend(
        name=name,
        slug=slug,
        prompts_dir=subdir,
        extension=extension,
        run_prefix=run_prefix,
        run_env_vars=_RUN_ENV_BY_SLUG.get(slug, {}),
    )


def discover_backends(prompts_dir: Path = PROMPTS_DIR) -> dict[str, Backend]:
    """Scan a prompts directory and return backends that have both required prompt files."""
    backends: dict[str, Backend] = {}
    if not prompts_dir.is_dir():
        return backends
    for subdir in sorted(prompts_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if (subdir / "translate.txt").exists() and (subdir / "fix_errors.txt").exists():
            backend = _build_backend(subdir)
            backends[backend.slug] = backend
    return backends


class BackendRegistry:
    """Lazy registry that discovers backends once and caches the result."""

    def __init__(self, prompts_dir: Path = PROMPTS_DIR) -> None:
        self._prompts_dir = prompts_dir
        self._cache: dict[str, Backend] | None = None

    def _load(self) -> dict[str, Backend]:
        if self._cache is None:
            self._cache = discover_backends(self._prompts_dir)
        return self._cache

    def all(self) -> dict[str, Backend]:
        """Return all discovered backends keyed by slug."""
        return self._load()

    def resolve(self, name: str) -> Backend:
        """Resolve a backend name (case-insensitive) to a Backend instance.

        Raises:
            ValueError: If the name does not match any known backend.
        """
        backends = self._load()
        key = name.lower()
        if key in backends:
            return backends[key]
        known = ", ".join(b.name for b in backends.values())
        raise ValueError(f"Unknown backend: {name!r}. Use one of: {known}")


# Module-level default registry used by CLI and tests.
_default_registry = BackendRegistry()


def resolve_backend(name: str) -> Backend:
    """Resolve a backend name using the default registry."""
    return _default_registry.resolve(name)


# ── Pipeline resolution ────────────────────────────────────────────────────────

def _resolve_prompt_file(
    filename: str,
    *,
    pipeline_stage_dir: Path | None,
    backend_dir: Path,
) -> Path | None:
    """Return the resolved path for a prompt file using pipeline-override fallback.

    Search order:
    1. ``<pipeline_stage_dir>/<filename>``  — pipeline-specific override
    2. ``<backend_dir>/<filename>``          — generic backend fallback

    Returns ``None`` if the file does not exist in either location.
    """
    if pipeline_stage_dir is not None:
        candidate = pipeline_stage_dir / filename
        if candidate.is_file():
            return candidate
    candidate = backend_dir / filename
    if candidate.is_file():
        return candidate
    return None


def _resolve_fix_execution(
    stage_name: str,
    *,
    pipeline_stage_dir: Path | None,
    backend_dir: Path,
) -> Path | None:
    """Resolve ``fix_execution_errors.txt``.

    Opt* stages use the file in their own directory, then the sibling
    translation stage of the same pipeline. They do not fall through to
    ``prompts/backends/``. Translation stages keep that fallback.
    """
    if pipeline_stage_dir is not None:
        own = pipeline_stage_dir / "fix_execution_errors.txt"
        if own.is_file():
            return own
    sibling = _OPT_FIX_SIBLING.get(stage_name.lower())
    if stage_name.lower().startswith("opt"):
        if sibling and pipeline_stage_dir is not None:
            candidate = pipeline_stage_dir.parent / sibling / "fix_execution_errors.txt"
            if candidate.is_file():
                logger.warning(
                    "  [WARNING] %s has no fix_execution_errors.txt; using %s",
                    stage_name,
                    candidate,
                )
                return candidate
        return None
    fallback = backend_dir / "fix_execution_errors.txt"
    if fallback.is_file():
        return fallback
    return None


def _resolve_stage_prompts(
    stage_name: str,
    backend: Backend,
    pipeline_stage_dir: Path | None,
) -> StagePrompts:
    """Build a StagePrompts for *stage_name* applying per-file fallback.

    Raises:
        ValueError: if ``translate.txt`` or ``fix_errors.txt`` cannot be resolved.
    """
    translate = _resolve_prompt_file(
        "translate.txt",
        pipeline_stage_dir=pipeline_stage_dir,
        backend_dir=backend.prompts_dir,
    )
    if translate is None:
        raise ValueError(
            f"translate.txt not found for stage {stage_name!r}. "
            f"Looked in {pipeline_stage_dir} and {backend.prompts_dir}."
        )

    fix_compile = _resolve_prompt_file(
        "fix_errors.txt",
        pipeline_stage_dir=pipeline_stage_dir,
        backend_dir=backend.prompts_dir,
    )
    if fix_compile is None:
        raise ValueError(
            f"fix_errors.txt not found for stage {stage_name!r}. "
            f"Looked in {pipeline_stage_dir} and {backend.prompts_dir}."
        )

    fix_execution = _resolve_fix_execution(
        stage_name,
        pipeline_stage_dir=pipeline_stage_dir,
        backend_dir=backend.prompts_dir,
    )

    return StagePrompts(
        translate=translate,
        fix_compile=fix_compile,
        fix_execution=fix_execution,
    )


def _find_pipeline_stage_dir(
    pipeline_dir: Path,
    stage_token: str,
) -> Path | None:
    """Locate the stage sub-directory inside *pipeline_dir* (case-insensitive)."""
    direct = pipeline_dir / stage_token
    if direct.is_dir():
        return direct
    for d in pipeline_dir.iterdir():
        if d.is_dir() and d.name.lower() == stage_token.lower():
            return d
    return None


def _accumulate_profile(
    current: ParadigmProfile,
    slug: str,
) -> ParadigmProfile:
    """Return the profile obtained by adding the paradigm contributed by *slug*.

    Only non-Opt translation stages update the profile.  Opt stages inherit the
    profile from the preceding stage (no new paradigm is added).
    """
    paradigm = _SLUG_TO_PARADIGM.get(slug)
    if paradigm == "mpi":
        return current.with_mpi()
    if paradigm == "openmp":
        return current.with_openmp()
    if paradigm == "avx":
        return current.with_avx()
    return current


def discover_pipelines() -> list[str]:
    """List available pipeline names discovered from ``prompts/pipelines/``."""
    if not PROMPTS_PIPELINES_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in PROMPTS_PIPELINES_DIR.iterdir()
        if d.is_dir()
    )


def resolve_pipeline(
    pipeline_name: str,
    backends_dir: Path = PROMPTS_BACKENDS_DIR,
    pipelines_dir: Path = PROMPTS_PIPELINES_DIR,
) -> Pipeline:
    """Resolve *pipeline_name* (case-insensitive) to a :class:`Pipeline`.

    Supports both legacy pipelines (``AVX-OpenMP-MPI``) and the new HPC
    top-down pipelines with ``Opt*`` stages
    (``OptSerial-MPI-OptMPI-OpenMP-OptOpenMP-AVX-OptAVX``).

    ``Opt*`` stage resolution rules:
    - Each ``Opt*`` stage inherits the backend of the immediately preceding
      stage (positional inheritance).
    - ``OptSerial`` (first position only) borrows the backend of the first
      *translation* stage for prompt fallback, but compiles with
      ``compile_variant="serial"`` (same flags as the serial baseline).
    - ``OptMPI``, ``OptOpenMP``, ``OptAVX`` are validated to match the paradigm
      of the immediately preceding translation stage.

    Raises:
        ValueError: if the pipeline directory is not found, a stage name does
                    not match a known backend or ``Opt*`` validation fails, or
                    required prompt files are missing after fallback.
    """
    # ── Case-insensitive directory match ──────────────────────────────────
    if not pipelines_dir.is_dir():
        raise ValueError(f"Pipelines directory not found: {pipelines_dir}")

    canonical_name: str | None = None
    pipeline_dir: Path | None = None
    for d in pipelines_dir.iterdir():
        if d.is_dir() and d.name.lower() == pipeline_name.lower():
            canonical_name = d.name
            pipeline_dir = d
            break

    if canonical_name is None or pipeline_dir is None:
        available = ", ".join(discover_pipelines()) or "(ninguno)"
        raise ValueError(
            f"Unknown pipeline: {pipeline_name!r}. "
            f"Available: {available}"
        )

    # ── Discover backends ─────────────────────────────────────────────────
    all_backends = discover_backends(backends_dir)

    # ── First pass: find the first translation backend (needed for OptSerial) ──
    stage_tokens = canonical_name.split("-")
    first_translation_backend: Backend | None = None
    for token in stage_tokens:
        if not token.lower().startswith("opt"):
            slug = token.lower()
            if slug in all_backends:
                first_translation_backend = all_backends[slug]
                break

    # ── Second pass: build stages ─────────────────────────────────────────
    stages: list[PipelineStage] = []
    accumulated_profile = ParadigmProfile()

    for i, token in enumerate(stage_tokens):
        slug = token.lower()
        is_opt = slug.startswith("opt")

        if is_opt:
            # ── Optimisation stage ─────────────────────────────────────
            if slug == "optserial":
                # OptSerial must be the first stage
                if i != 0:
                    raise ValueError(
                        f"'OptSerial' can only be the first stage of pipeline "
                        f"'{canonical_name}', but it is at position {i}."
                    )
                if first_translation_backend is None:
                    raise ValueError(
                        f"Pipeline '{canonical_name}' has OptSerial at position 0 "
                        f"but contains no later translation stage."
                    )
                backend = first_translation_backend
                compile_variant = "serial"
                # OptSerial produces serial-optimised code; no paradigm added.
                stage_profile = ParadigmProfile()

            else:
                # OptMPI / OptOpenMP / OptAVX — must follow the right paradigm
                if i == 0:
                    raise ValueError(
                        f"Stage '{token}' cannot be the first stage of pipeline "
                        f"'{canonical_name}' (only OptSerial can)."
                    )
                backend = stages[i - 1].backend  # positional inheritance

                # Validate paradigm correspondence
                expected_paradigm = _OPT_SUFFIX_TO_PARADIGM.get(slug)
                if expected_paradigm is not None:
                    prev_backend_slug = stages[i - 1].backend.slug
                    prev_paradigm = _SLUG_TO_PARADIGM.get(prev_backend_slug)
                    if prev_paradigm != expected_paradigm:
                        raise ValueError(
                            f"Stage '{token}' must follow a stage of paradigm "
                            f"'{expected_paradigm}', but the previous stage uses backend "
                            f"'{stages[i - 1].backend.name}' (paradigm: "
                            f"'{prev_paradigm or 'none'}')."
                        )

                compile_variant = "translated"
                # Opt inherits the accumulated profile without adding new paradigm.
                stage_profile = accumulated_profile

        else:
            # ── Translation stage ──────────────────────────────────────
            if slug not in all_backends:
                known = ", ".join(b.name for b in all_backends.values())
                raise ValueError(
                    f"Stage {token!r} of pipeline {canonical_name!r} does not "
                    f"match any known backend. Available: {known}"
                )
            backend = all_backends[slug]
            compile_variant = "translated"
            # Accumulate paradigm from this translation stage.
            accumulated_profile = _accumulate_profile(accumulated_profile, slug)
            stage_profile = accumulated_profile

        # ── Resolve prompts ───────────────────────────────────────────
        pipeline_stage_dir = _find_pipeline_stage_dir(pipeline_dir, token)
        prompts = _resolve_stage_prompts(token, backend, pipeline_stage_dir)

        stages.append(
            PipelineStage(
                backend=backend,
                prompts=prompts,
                stage_index=i,
                is_last=(i == len(stage_tokens) - 1),
                logical_name=token,
                is_optimization=is_opt,
                compile_variant=compile_variant,
                profile=stage_profile,
            )
        )

    return Pipeline(name=canonical_name, stages=tuple(stages))
