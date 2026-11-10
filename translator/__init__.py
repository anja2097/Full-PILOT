"""Traductor serial → backend paralelo (OpenMP, Kokkos, MPI, Vectorize, …)."""

# Domain types are always available (no external dependencies).
from translator.domain.pipeline import Pipeline, PipelineStage, StagePrompts

# Services and CLI imports require optional runtime dependencies (pydantic, etc.).
# Wrap them so that tests and partial environments can still import domain types.
try:
    from translator.cli.main import main
    from translator.services.pipeline_runner import run_pipeline
    from translator.services.translate import translate_file
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    main = None            # type: ignore[assignment]
    run_pipeline = None    # type: ignore[assignment]
    translate_file = None  # type: ignore[assignment]

__all__ = [
    "main",
    "translate_file",
    "run_pipeline",
    "Pipeline",
    "PipelineStage",
    "StagePrompts",
]
