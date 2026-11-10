"""Use case: translate a serial source file to a parallel backend with iterative error fixing."""

from __future__ import annotations

import logging
from pathlib import Path

from translator.config.settings import API_URL, CHECKSUM_RTOL
from translator.domain.backend import Backend
from translator.infrastructure.execution.benchmark import verify_and_benchmark
from translator.services.metrics import MetricsAccumulator, print_metrics
from translator.services.stage_runner import (
    LLMConfig,
    run_serial_reference,
    run_translation_stage,
)

logger = logging.getLogger(__name__)


def translate_file(
    source_path: Path,
    api_key: str,
    *,
    backend: Backend,
    model: str,
    api_url: str = API_URL,
    model_provider: str | None = None,
    thinking: bool = False,
    thinking_effort: str | None = None,
    extra_flags: list[str] | None = None,
    compare: bool = True,
    metrics: bool = False,
    rtol: float = CHECKSUM_RTOL,
) -> None:
    """Translate a serial source file to the given backend, fixing errors iteratively.

    When *compare* is True the loop also verifies execution correctness: after each
    successful compilation the translated binary is run and its output is compared to
    the serial reference.  If the result is outside tolerance, ``fix_execution_errors.txt``
    is used to ask the LLM to fix runtime errors.  The total number of LLM iterations
    (compilation fixes + execution fixes) is bounded by MAX_RETRIES.
    """
    translated_path = backend.translated_path(source_path)

    llm_cfg = LLMConfig(
        model=model,
        api_url=api_url,
        model_provider=model_provider,
        thinking=thinking,
        thinking_effort=thinking_effort,
    )

    acc: MetricsAccumulator | None = MetricsAccumulator() if metrics else None

    serial_stdout: str = ""
    serial_checksum: float | None = None
    serial_checksum_text: str | None = None

    if compare:
        serial_stdout, serial_checksum, serial_checksum_text = run_serial_reference(
            source_path, backend, extra_flags=extra_flags,
        )

    fix_exec_path = backend.prompts_dir / "fix_execution_errors.txt"
    if not fix_exec_path.is_file():
        fix_exec_path = None  # type: ignore[assignment]

    try:
        run_translation_stage(
            source_path,
            translated_path,
            backend=backend,
            translate_prompt_path=backend.prompts_dir / "translate.txt",
            fix_compile_prompt_path=backend.prompts_dir / "fix_errors.txt",
            fix_exec_prompt_path=fix_exec_path,
            llm_cfg=llm_cfg,
            api_key=api_key,
            extra_flags=extra_flags,
            compare=compare,
            serial_stdout=serial_stdout,
            serial_checksum=serial_checksum,
            serial_checksum_text=serial_checksum_text,
            acc=acc,
            rtol=rtol,
        )
        verify_and_benchmark(source_path, backend, extra_flags=extra_flags, rtol=rtol)
    finally:
        if acc is not None and acc.calls:
            acc.resolve_generation_metadata(api_key)
            print_metrics(acc)
