"""Core translation stage executor shared by single-backend and pipeline modes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from translator.config.settings import API_URL, CHECKSUM_RTOL, MAX_RETRIES
from translator.domain.backend import Backend
from translator.exceptions import MaxRetriesExhaustedError
from translator.infrastructure.compilation import compile_file, executable_path
from translator.infrastructure.execution.benchmark import run_and_check
from translator.infrastructure.execution.output_parser import parse_checksum
from translator.infrastructure.execution.process import run_once
from translator.infrastructure.llm.client import chat_completion
from translator.infrastructure.llm.code_extractor import (
    extract_content,
    extract_message,
    strip_analysis_comment,
)
from translator.infrastructure.llm.models import Message
from translator.services.metrics import MetricsAccumulator
from translator.services.prompt_blocks import compose_stage_prompt

if TYPE_CHECKING:
    from translator.domain.pipeline import ParadigmProfile
    from translator.infrastructure.hardware import HardwareInfo, MpiLaunch

logger = logging.getLogger(__name__)


class LLMConfig(NamedTuple):
    """Parameters forwarded to every chat_completion() call."""

    model: str
    api_url: str = API_URL
    model_provider: str | None = None
    thinking: bool = False
    thinking_effort: str | None = None


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assistant_turn(message: Message) -> dict:
    turn: dict = {
        "role": "assistant",
        "content": message.content,
    }
    if message.reasoning_details is not None:
        turn["reasoning_details"] = [rd.model_dump() for rd in message.reasoning_details]
    return turn


def _write_source(target_path: Path, code: str) -> None:
    target_path.write_text(code, encoding="utf-8")
    logger.info("  File updated: %s", target_path)


def run_serial_reference(
    source_path: Path,
    backend: Backend,
    *,
    extra_flags: list[str] | None = None,
) -> tuple[str, float | None, str | None]:
    """Compile and run the serial source to obtain stdout and checksum for reference.

    Returns:
        (serial_stdout, serial_checksum, printed_token). Checksum and token are
        None when the program prints no checksum.

    Raises:
        CompilationError: if the serial source fails to compile.
    """
    from translator.exceptions import CompilationError

    logger.info("\n\n  Compiling the serial version for the reference…")
    ok, err = compile_file(source_path, backend, variant="serial", extra_flags=extra_flags)
    if not ok:
        raise CompilationError("Could not compile the original file:", output=err)

    serial_exe = executable_path(source_path)
    serial_cmd = backend.run_serial_cmd(serial_exe)
    logger.info("  Running the serial reference…")
    serial_stdout, _serial_stderr, _elapsed = run_once(serial_exe, serial_cmd)
    if serial_exe.is_file():
        serial_exe.unlink()

    serial_checksum, serial_checksum_text = parse_checksum(serial_stdout)
    if serial_checksum is not None:
        logger.info("  Serial reference checksum: %.6g", serial_checksum)
    else:
        logger.info("  (Checksum unavailable in the serial run — stdout will be compared exactly)")
    return serial_stdout, serial_checksum, serial_checksum_text


def run_translation_stage(
    input_path: Path,
    output_path: Path,
    *,
    backend: Backend,
    translate_prompt_path: Path,
    fix_compile_prompt_path: Path,
    fix_exec_prompt_path: Path | None,
    llm_cfg: LLMConfig,
    api_key: str,
    extra_flags: list[str] | None = None,
    compare: bool = True,
    serial_stdout: str = "",
    serial_checksum: float | None = None,
    serial_checksum_text: str | None = None,
    profile: "ParadigmProfile | None" = None,
    mpi_backend: Backend | None = None,
    hardware: "HardwareInfo | None" = None,
    mpi_launch: "MpiLaunch | None" = None,
    compile_variant: str = "translated",
    # Legacy flags kept for backward compatibility with single-backend callers.
    # Ignored when *profile* is provided.
    hybrid_openmp: bool = False,
    hybrid_avx: bool = False,
    acc: MetricsAccumulator | None = None,
    stage_label: str = "",
    logical_name: str = "",
    rtol: float = CHECKSUM_RTOL,
    performance: str | None = None,
    vec_report: str | None = None,
    audit_model: str | None = None,
) -> None:
    """Translate *input_path* to *output_path* for *backend*, fixing errors iteratively.

    This function is the shared core used by both ``translate_file`` (single backend)
    and ``pipeline_runner`` (multi-stage pipelines).

    Args:
        input_path:              Source file to translate.
        output_path:             Destination file for the translated code.
        backend:                 Backend whose toolchain is used for compilation.
        translate_prompt_path:   Path to the translation prompt file.
        fix_compile_prompt_path: Path to the compile-error fix prompt.
        fix_exec_prompt_path:    Path to the execution-error fix prompt, or ``None``
                                 to disable compare-mode for this stage.
        llm_cfg:                 LLM API parameters.
        api_key:                 API key for the LLM provider.
        extra_flags:             Extra compiler flags.
        compare:                 Whether to verify execution correctness in the retry loop.
        serial_stdout:           Reference stdout from the original serial run.
        serial_checksum:         Reference checksum from the original serial run.
        profile:                 Accumulated paradigm profile.  When provided, determines
                                 the compiler and runner for this stage.  Overrides
                                 ``hybrid_openmp`` / ``hybrid_avx``.
        mpi_backend:             MPI backend used when ``profile.mpi`` is True.
        compile_variant:         ``"serial"`` for OptSerial; ``"translated"`` for all others.
        hybrid_openmp:           Legacy flag — ignored when *profile* is provided.
        hybrid_avx:              Legacy flag — ignored when *profile* is provided.
        acc:                     Shared MetricsAccumulator; pass ``None`` to skip metrics.
        stage_label:             Human-readable prefix for LLM call labels.

    Raises:
        MaxRetriesExhaustedError: if the code cannot be fixed within MAX_RETRIES.
    """
    if fix_exec_prompt_path is None and compare:
        logger.warning(
            "  [WARNING] --compare is on but fix_execution_errors.txt was not found for %s. "
            "Execution verification is disabled for this stage.",
            stage_label or backend.name,
        )
        compare = False

    from translator.infrastructure.hardware import include_mpi_launch_prompt

    translate_prompt = compose_stage_prompt(
        _load_prompt(translate_prompt_path),
        hardware=hardware,
        mpi_launch=mpi_launch,
        include_mpi_launch=include_mpi_launch_prompt(backend.slug, compile_variant),
        logical_name=logical_name,
        profile_mpi=bool(profile is not None and profile.mpi),
        rtol=rtol,
        performance=performance,
        vec_report=vec_report,
    )
    fix_compile_prompt = _load_prompt(fix_compile_prompt_path)
    fix_exec_prompt: str | None = (
        _load_prompt(fix_exec_prompt_path) if fix_exec_prompt_path else None
    )

    # Determine effective hybrid flags for legacy path (when profile is None)
    if profile is None:
        eff_hybrid_openmp = hybrid_openmp
        eff_hybrid_avx = hybrid_avx
    else:
        eff_hybrid_openmp = False
        eff_hybrid_avx = False

    source_code = strip_analysis_comment(input_path.read_text(encoding="utf-8"))
    logger.info("\n\n  Original:  %s", input_path)
    logger.info("  Translated: %s\n", output_path)

    user_msg = {"role": "user", "content": f"{translate_prompt}\n\n{source_code}"}
    messages: list[dict] = [user_msg]

    def _iter_label(iteration: int, kind: str = "") -> str:
        base = f"{stage_label} — " if stage_label else ""
        suffix = f" — {kind}" if kind else ""
        return f"{base}Iteration {iteration}{suffix}"

    def _audit_translation() -> tuple[bool, str]:
        """Audit a compiled (and, when requested, verified) translation."""
        if not audit_model:
            return True, ""
        from translator.services.mol_runner import _audit

        passed, reason, data = _audit(
            original_code=source_code,
            optimized_code=output_path.read_text(encoding="utf-8"),
            audit_model=audit_model,
            api_key=api_key,
            llm_cfg=llm_cfg,
            stage_label=stage_label,
            rtol=rtol,
            mode="translation",
        )
        if acc is not None and data is not None:
            acc.add(f"{stage_label} — Translation auditor", data)
        if passed:
            return True, ""
        shown = reason or "CHEATS"
        return False, (
            f"The auditor rejected this translation: {shown}. "
            "Fix it so the program performs exactly the work of the original timed kernel."
        )

    def _accept_or_audit(iteration: int, *, verified: bool) -> bool:
        """Return True when the iteration is finished and accepted."""
        audit_ok, audit_error = _audit_translation()
        if audit_ok:
            if verified:
                logger.info(
                    "\n\nCompiled and verified successfully on iteration %s.",
                    iteration,
                )
            else:
                logger.info("\n\nCompiled successfully on iteration %s.", iteration)
            return True
        nonlocal_error.append(audit_error)
        return False

    # ── Iteración 1: traducción inicial ─────────────────────────────────────
    label1 = _iter_label(1, "initial translation")
    data = chat_completion(
        api_key,
        messages,
        label=label1,
        model=llm_cfg.model,
        api_url=llm_cfg.api_url,
        model_provider=llm_cfg.model_provider,
        thinking=llm_cfg.thinking,
        thinking_effort=llm_cfg.thinking_effort,
    )
    if acc is not None:
        acc.add(label1, data)

    assistant_msg = extract_message(data)
    messages.append(_assistant_turn(assistant_msg))

    translated_code = extract_content(data)
    _write_source(output_path, translated_code)

    nonlocal_error: list[str] = []

    ok, compile_error = compile_file(
        output_path, backend,
        variant=compile_variant,
        extra_flags=extra_flags,
        profile=profile,
        hybrid_openmp=eff_hybrid_openmp,
        hybrid_avx=eff_hybrid_avx,
    )

    if ok:
        exec_ok, exec_error = True, ""
        if compare:
            exec_ok, exec_error = run_and_check(
                output_path, backend,
                serial_stdout=serial_stdout,
                serial_checksum=serial_checksum,
                extra_flags=extra_flags,
                translated_path=output_path,
                profile=profile,
                mpi_backend=mpi_backend,
                hardware=hardware,
                mpi_launch=mpi_launch,
                compile_variant=compile_variant,
                rtol=rtol,
                serial_checksum_text=serial_checksum_text,
            )
        if not compare or exec_ok:
            if _accept_or_audit(1, verified=bool(compare and exec_ok)):
                return
            error_type = "execution"
            last_error = nonlocal_error[-1]
        else:
            error_type = "execution"
            last_error = exec_error
    else:
        error_type = "compilation"
        last_error = compile_error

    # ── Iteraciones 2..MAX_RETRIES ───────────────────────────────────────────
    for iteration in range(2, MAX_RETRIES + 1):
        current_code = output_path.read_text(encoding="utf-8")

        if error_type == "compilation":
            fix_content = (
                f"{fix_compile_prompt}\n\n{current_code}"
                f"\n\nCompilation errors:\n{last_error}"
            )
            kind = "compilation fix"
        else:
            if not fix_exec_prompt:
                raise MaxRetriesExhaustedError(
                    "No fix_execution_errors.txt to correct the execution "
                    f"or audit failure.\nLast error:\n{last_error}"
                )
            fix_content = (
                f"{fix_exec_prompt}\n\n**Current code:**\n\n{current_code}"
                f"\n\n**Execution error details:**\n{last_error}"
            )
            kind = "execution-error fix"

        label_n = _iter_label(iteration, kind)
        messages.append({"role": "user", "content": fix_content})

        data = chat_completion(
            api_key,
            messages,
            label=label_n,
            model=llm_cfg.model,
            api_url=llm_cfg.api_url,
            model_provider=llm_cfg.model_provider,
            thinking=llm_cfg.thinking,
            thinking_effort=llm_cfg.thinking_effort,
        )
        if acc is not None:
            acc.add(label_n, data)

        assistant_msg = extract_message(data)
        messages.append(_assistant_turn(assistant_msg))

        translated_code = extract_content(data)
        _write_source(output_path, translated_code)

        ok, compile_error = compile_file(
            output_path, backend,
            variant=compile_variant,
            extra_flags=extra_flags,
            profile=profile,
            hybrid_openmp=eff_hybrid_openmp,
            hybrid_avx=eff_hybrid_avx,
        )
        if not ok:
            error_type = "compilation"
            last_error = compile_error
            continue

        if not compare:
            if _accept_or_audit(iteration, verified=False):
                return
            error_type = "execution"
            last_error = nonlocal_error[-1]
            continue

        exec_ok, exec_error = run_and_check(
            output_path, backend,
            serial_stdout=serial_stdout,
            serial_checksum=serial_checksum,
            extra_flags=extra_flags,
            translated_path=output_path,
            profile=profile,
            mpi_backend=mpi_backend,
            hardware=hardware,
            mpi_launch=mpi_launch,
            compile_variant=compile_variant,
            rtol=rtol,
            serial_checksum_text=serial_checksum_text,
        )
        if exec_ok:
            if _accept_or_audit(iteration, verified=True):
                return
            error_type = "execution"
            last_error = nonlocal_error[-1]
            continue

        error_type = "execution"
        last_error = exec_error

    if error_type == "compilation":
        raise MaxRetriesExhaustedError(
            f"Could not compile after {MAX_RETRIES} iterations. "
            f"Last error:\n{last_error}"
        )
    raise MaxRetriesExhaustedError(
        f"Could not correct the execution result after {MAX_RETRIES} iterations. "
        f"Last error:\n{last_error}"
    )
