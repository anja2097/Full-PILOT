"""Mix of LLMs (MoL) optimization stage runner."""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from translator.config.settings import (
    CHECKSUM_RTOL,
    EXECUTION_TIMEOUT_SECONDS,
    MOL_FIX_ATTEMPTS,
    MOL_SPEEDUP_AUDIT_THRESHOLD,
    format_tolerance,
)
from translator.domain.backend import Backend
from translator.infrastructure.compilation import compile_file, executable_path
from translator.infrastructure.execution.benchmark import benchmark_stage, effective_run
from translator.exceptions import ExecutionError, ExecutionTimeoutError
from translator.infrastructure.execution.output_parser import (
    absolute_tolerance,
    checksums_match,
    first_stdout_diff,
    parse_checksum,
)
from translator.infrastructure.execution.process import run_once
from translator.infrastructure.llm.client import chat_completion
from translator.infrastructure.llm.code_extractor import (
    extract_content,
    extract_message,
    strip_analysis_comment,
)
from translator.infrastructure.llm.models import ChatCompletion, Message
from translator.services.metrics import MetricsAccumulator
from translator.services.stage_runner import LLMConfig

if TYPE_CHECKING:
    from translator.domain.pipeline import ParadigmProfile
    from translator.infrastructure.hardware import HardwareInfo, MpiLaunch

logger = logging.getLogger(__name__)


def _assistant_turn(message: Message) -> dict:
    """One assistant turn, including reasoning details when the model returned them."""
    turn: dict = {
        "role": "assistant",
        "content": message.content,
    }
    if message.reasoning_details is not None:
        turn["reasoning_details"] = [rd.model_dump() for rd in message.reasoning_details]
    return turn


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class MoLConfig:
    """Configuration for Mix of LLMs optimization."""

    agent_models: list[str]
    audit_model: str
    speedup_audit_threshold: float = MOL_SPEEDUP_AUDIT_THRESHOLD


@dataclass
class _Variant:
    """Internal state for a single MoL candidate."""

    agent_index: int
    model: str
    code: str
    # Populated during sequential evaluation:
    compiled: bool = False
    correct: bool = False
    wall_ms: float | None = None
    reported_ms: float | None = None
    audit_passed: bool = False
    thread_config: str | None = None
    launch_line: str = ""
    # Deferred metrics (added to acc in main thread):
    _llm_results: list[tuple[str, ChatCompletion]] = field(default_factory=list)


# ── Public entry point ────────────────────────────────────────────────────────

def run_optimization_stage(
    input_path: Path,
    output_path: Path,
    *,
    backend: Backend,
    compile_variant: str = "translated",
    translate_prompt: str,
    fix_compile_prompt: str,
    fix_exec_prompt: str = "",
    llm_cfg: LLMConfig,
    mol_cfg: MoLConfig,
    api_key: str,
    extra_flags: list[str] | None = None,
    serial_stdout: str = "",
    serial_checksum: float | None = None,
    serial_checksum_text: str | None = None,
    profile: "ParadigmProfile | None" = None,
    mpi_backend: Backend | None = None,
    hardware: "HardwareInfo | None" = None,
    mpi_launch: "MpiLaunch | None" = None,
    acc: MetricsAccumulator | None = None,
    stage_label: str = "",
    debug: bool = False,
    baseline_wall_ms: float | None = None,
    baseline_reported_ms: float | None = None,
    logical_name: str = "",
    rtol: float = CHECKSUM_RTOL,
) -> tuple[bool, float | None, float | None, str | None, str]:
    """Execute an optimization stage using Mix of LLMs.

    Returns:
        ``(accepted, wall_ms, reported_ms, thread_config, launch_line)``.
        ``accepted`` is True when a winning variant was written to
        *output_path*. The two times are that variant's benchmark from before
        the audit (process time and kernel time). When the input is kept, the
        result is ``(False, None, None, None, "")``.

    Flow:
    1. Reuse a baseline already measured by the caller, or benchmark the input.
    2. Fan-out: call N agent LLMs in parallel threads (LLM calls only).
    3. For each variant sequentially: compile (fix turns keep that agent's
       conversation), verify correctness, benchmark.
       Failures are isolated per-agent.
    4. Rank valid variants by reported time when both sides have it, else by
       process time.
    5. Audit only candidates that clear ``speedup_audit_threshold``.
    6. Accept the first audited variant that passes. Otherwise keep the input.
    7. Recompile the winner in *output_path* so the next pipeline stage can
       rely on the binary being present.

    Args:
        input_path:        Code from the previous stage (to optimize).
        output_path:       Where to write the optimized code.
        backend:           Backend for compilation (inherited from the preceding
                           pipeline stage).
        compile_variant:   ``"serial"`` for OptSerial; ``"translated"`` for others.
        translate_prompt:  Optimization prompt (hardware info already prepended).
        fix_compile_prompt: Prompt for compile-error fixes.
        llm_cfg:           Base LLM config (used for the auditor).
        mol_cfg:           MoL-specific configuration.
        api_key:           API key.
        extra_flags:       Extra compiler flags.
        serial_stdout:     Reference stdout from the serial run.
        serial_checksum:   Reference checksum from the serial run.
        profile:           Accumulated paradigm profile for compile/run dispatch.
        mpi_backend:       MPI backend for execution when profile.mpi is True.
        hardware:          Host topology used to size mpirun / OpenMP.
        acc:               Metrics accumulator (thread-safe: only written from the
                           main thread after parallel LLM calls complete).
        stage_label:       Label for logging.
        debug:             Keep ``*_mol_vN`` sources on disk.
        baseline_wall_ms:  Process time already measured for this input.
        baseline_reported_ms: Kernel time already measured for this input.
    """
    source_code = strip_analysis_comment(input_path.read_text(encoding="utf-8"))

    # ── 1. Baseline already measured, or benchmark the input ──────────────
    if baseline_wall_ms is None:
        baseline_wall_ms, measured_reported = _benchmark_input(
            input_path, backend,
            compile_variant=compile_variant,
            profile=profile, mpi_backend=mpi_backend,
            hardware=hardware,
            mpi_launch=mpi_launch,
            extra_flags=extra_flags,
            stage_label=stage_label,
            logical_name=logical_name,
        )
        if baseline_reported_ms is None:
            baseline_reported_ms = measured_reported
    else:
        logger.info(
            "\n  [MoL] Reused baseline: process %.1f ms, reported %s",
            baseline_wall_ms,
            f"{baseline_reported_ms:.1f} ms" if baseline_reported_ms is not None else "N/A",
        )

    # ── 2. Fan-out: parallel LLM generation ──────────────────────────────
    logger.info("\n  [MoL] Launching %s agents in parallel…", len(mol_cfg.agent_models))
    agent_responses = _parallel_generate(
        source_code=source_code,
        translate_prompt=translate_prompt,
        api_key=api_key,
        llm_cfg=llm_cfg,
        mol_cfg=mol_cfg,
        stage_label=stage_label,
    )

    # ── 3. Sequential evaluation: compile → verify → benchmark ───────────
    variants: list[_Variant] = []
    for agent_index, model, code, messages, llm_results in agent_responses:
        variant = _Variant(agent_index=agent_index, model=model, code=code)
        variant._llm_results.extend(llm_results)

        variant_path = _variant_path(output_path, agent_index)
        variant_path.write_text(code, encoding="utf-8")

        ready, final_code, fix_results = _correct_and_verify(
            variant_path, backend, model,
            fix_compile_prompt, fix_exec_prompt,
            api_key, llm_cfg, extra_flags, profile,
            messages=messages,
            compile_variant=compile_variant,
            stage_label=stage_label, agent_idx=agent_index,
            serial_stdout=serial_stdout,
            serial_checksum=serial_checksum,
            serial_checksum_text=serial_checksum_text,
            mpi_backend=mpi_backend,
            hardware=hardware,
            mpi_launch=mpi_launch,
            rtol=rtol,
        )
        variant._llm_results.extend(fix_results)

        if not ready:
            _cleanup_variant(variant_path, keep_source=debug)
            variants.append(variant)
            continue

        variant.compiled = True
        variant.correct = True
        variant.code = final_code
        variant_path.write_text(final_code, encoding="utf-8")

        # Benchmark
        wall_ms, reported_ms, thread_config, launch_line = _bench_variant(
            variant_path, backend,
            profile=profile, mpi_backend=mpi_backend,
            hardware=hardware,
            mpi_launch=mpi_launch,
            compile_variant=compile_variant,
            stage_label=stage_label, agent_index=agent_index,
            logical_name=logical_name,
        )
        variant.wall_ms = wall_ms
        variant.reported_ms = reported_ms
        variant.thread_config = thread_config
        variant.launch_line = launch_line
        metric, metric_ms = selection_metric(
            baseline_reported_ms, reported_ms, wall_ms,
        )
        baseline_for_log = (
            baseline_reported_ms if metric == "reported" else baseline_wall_ms
        )
        config = f", config {thread_config}" if thread_config else ""
        logger.info(
            "  [MoL] Agent %d: %.1f ms (metric %s%s, baseline: %s)",
            agent_index + 1,
            metric_ms if metric_ms is not None else wall_ms,
            metric,
            config,
            f"{baseline_for_log:.1f} ms" if baseline_for_log is not None else "N/A",
        )

        _cleanup_variant(variant_path, keep_source=debug)
        variants.append(variant)

    # Flush LLM metrics to accumulator (main thread — not inside threads)
    if acc is not None:
        for v in variants:
            for label, data in v._llm_results:
                acc.add(label, data)

    # ── 4. Rank valid variants ─────────────────────────────────────────────
    valid = [v for v in variants if v.correct and v.wall_ms is not None]
    if not valid:
        logger.info("\n  [MoL] No valid variants. Keeping the input unchanged.")
        shutil.copy2(input_path, output_path)
        _recompile_output(output_path, backend, compile_variant, extra_flags, profile)
        return False, None, None, None, ""

    # ── 5. Audit only candidates that clear the margin ────────────────────
    scored = _scored_candidates(valid, baseline_wall_ms, baseline_reported_ms)
    ranked = ordered_audit_candidates(
        valid, baseline_wall_ms, baseline_reported_ms, mol_cfg.speedup_audit_threshold,
    )
    below = [item for item in scored if item[3] < mol_cfg.speedup_audit_threshold]
    for candidate, metric, value, speedup in below:
        logger.info(
            "  [MoL] Agent %d does not clear the margin (metric %s, %.1f ms, speedup %.3fx). Not audited.",
            candidate.agent_index + 1, metric, value, speedup,
        )

    winner: _Variant | None = None
    winner_metric = "process"
    winner_ms: float | None = None
    audit_llm_results: list[tuple[str, ChatCompletion]] = []

    if not ranked:
        logger.info(
            "\n  [MoL] No variant improves the baseline by at least %.0f%%. Keeping the input.",
            (mol_cfg.speedup_audit_threshold - 1.0) * 100.0,
        )
        shutil.copy2(input_path, output_path)
        _recompile_output(output_path, backend, compile_variant, extra_flags, profile)
        return False, None, None, None, ""

    for candidate, metric, value, speedup in ranked:
        logger.info(
            "\n  [MoL] Auditing agent %d (%.1f ms, metric %s, speedup %.2fx)…",
            candidate.agent_index + 1, value, metric, speedup,
        )

        ok, _reason, audit_data = _audit(
            original_code=source_code,
            optimized_code=candidate.code,
            audit_model=mol_cfg.audit_model,
            api_key=api_key,
            llm_cfg=llm_cfg,
            stage_label=stage_label,
            rtol=rtol,
        )
        if audit_data is not None:
            audit_llm_results.append((f"{stage_label} — MoL Audit", audit_data))

        if ok:
            candidate.audit_passed = True
            winner = candidate
            winner_metric = metric
            winner_ms = value
            logger.info("  [MoL] Audit PASSED.")
            break
        else:
            logger.warning("  [MoL] Audit REJECTED. Trying the next candidate…")

    if acc is not None:
        for label, data in audit_llm_results:
            acc.add(label, data)

    if winner is None:
        logger.info("\n  [MoL] All candidates failed the audit. Keeping the input.")
        shutil.copy2(input_path, output_path)
        _recompile_output(output_path, backend, compile_variant, extra_flags, profile)
        return False, None, None, None, ""

    # ── 6. Write winner and recompile ─────────────────────────────────────
    output_path.write_text(winner.code, encoding="utf-8")
    logger.info(
        "\n  [MoL] Winning variant: agent %d (%s) — %.1f ms (metric %s)",
        winner.agent_index + 1, winner.model, winner_ms, winner_metric,
    )
    _recompile_output(output_path, backend, compile_variant, extra_flags, profile)
    return True, winner.wall_ms, winner.reported_ms, winner.thread_config, winner.launch_line


# ── Internal helpers ──────────────────────────────────────────────────────────

def _variant_path(output_path: Path, agent_index: int) -> Path:
    return output_path.with_name(
        f"{output_path.stem}_mol_v{agent_index}{output_path.suffix}"
    )


def selection_metric(
    baseline_reported: float | None,
    candidate_reported: float | None,
    candidate_wall: float | None,
) -> tuple[str, float | None]:
    """``reported`` when both baseline and candidate have it; otherwise ``process``."""
    if baseline_reported is not None and candidate_reported is not None:
        return "reported", candidate_reported
    return "process", candidate_wall


def _scored_candidates(
    valid: list[_Variant],
    baseline_wall: float | None,
    baseline_reported: float | None,
) -> list[tuple[_Variant, str, float, float]]:
    """Candidatos con tiempo, del más rápido al más lento, y su speedup."""
    scored: list[tuple[_Variant, str, float, float]] = []
    for variant in valid:
        label, value = selection_metric(
            baseline_reported, variant.reported_ms, variant.wall_ms,
        )
        if value is None or value <= 0:
            continue
        base = baseline_reported if label == "reported" else baseline_wall
        speedup = (base / value) if base is not None and base > 0 else 0.0
        scored.append((variant, label, value, speedup))
    scored.sort(key=lambda item: item[2])
    return scored


def ordered_audit_candidates(
    valid: list[_Variant],
    baseline_wall: float | None,
    baseline_reported: float | None,
    min_speedup: float,
) -> list[tuple[_Variant, str, float, float]]:
    """Candidatos que superan el margen, en el orden en que se auditan."""
    return [
        item
        for item in _scored_candidates(valid, baseline_wall, baseline_reported)
        if item[3] >= min_speedup
    ]


def _cleanup_variant(variant_path: Path, *, keep_source: bool = False) -> None:
    """Remove the variant binary and, unless debugging, its source."""
    if variant_path.is_file() and not keep_source:
        variant_path.unlink()
    exe = executable_path(variant_path)
    if exe.is_file():
        exe.unlink()


def _parallel_generate(
    *,
    source_code: str,
    translate_prompt: str,
    api_key: str,
    llm_cfg: LLMConfig,
    mol_cfg: MoLConfig,
    stage_label: str,
) -> list[tuple[int, str, str, list[dict], list[tuple[str, ChatCompletion]]]]:
    """Call each agent model concurrently.

    Returns ``(idx, model, code, messages, llm_results)`` in agent order.
    ``messages`` is that agent's own conversation: the optimisation prompt and
    its reply, including reasoning details when present.  Failures are caught
    per-agent so a timeout or empty response does not abort the stage.
    """
    def _call(
        agent_index: int, model: str,
    ) -> tuple[int, str, str, list[dict], list[tuple[str, ChatCompletion]]]:
        label = f"{stage_label} — MoL Agent {agent_index + 1} ({model})"
        messages: list[dict] = [
            {"role": "user", "content": f"{translate_prompt}\n\n{source_code}"},
        ]
        try:
            data = chat_completion(
                api_key, messages, label=label,
                model=model,
                api_url=llm_cfg.api_url,
                model_provider=llm_cfg.model_provider,
                thinking=llm_cfg.thinking,
                thinking_effort=llm_cfg.thinking_effort,
            )
            code = extract_content(data)
            messages.append(_assistant_turn(extract_message(data)))
            return agent_index, model, code, messages, [(label, data)]
        except Exception as exc:
            logger.warning(
                "  [MoL] Agent %d (%s) failed the LLM call: %s",
                agent_index + 1, model, exc,
            )
            return agent_index, model, "", [], []

    results: list[tuple[int, str, str, list[dict], list[tuple[str, ChatCompletion]]]] = []
    models = mol_cfg.agent_models
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models) or 1) as executor:
        futures = {
            executor.submit(_call, index, model): (index, model)
            for index, model in enumerate(models)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover
                idx, model = futures[future]
                logger.warning("  [MoL] Agent %d: unexpected error: %s", idx + 1, exc)
                results.append((idx, model, "", [], []))

    # Restore deterministic order
    results.sort(key=lambda t: t[0])
    # Skip agents that returned empty code
    return [(i, m, c, msgs, lr) for i, m, c, msgs, lr in results if c.strip()]


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _discard(agent_idx: int) -> None:
    logger.info("  [MoL] Agent %d: discarded.", agent_idx + 1)


def _request_fix(
    *,
    messages: list[dict],
    content: str,
    api_key: str,
    model: str,
    llm_cfg: LLMConfig,
    label: str,
    llm_results: list[tuple[str, ChatCompletion]],
    variant_path: Path,
    agent_idx: int,
    fix_number: int,
) -> bool:
    """Append one correction turn. Return False when the model call fails."""
    messages.append({"role": "user", "content": content})
    try:
        data = chat_completion(
            api_key, messages,
            label=label, model=model,
            api_url=llm_cfg.api_url,
            model_provider=llm_cfg.model_provider,
            thinking=llm_cfg.thinking,
            thinking_effort=llm_cfg.thinking_effort,
        )
    except Exception as exc:
        logger.warning(
            "  [MoL] Agent %d fix %d failed: %s",
            agent_idx + 1, fix_number, exc,
        )
        return False
    llm_results.append((label, data))
    messages.append(_assistant_turn(extract_message(data)))
    variant_path.write_text(extract_content(data), encoding="utf-8")
    return True


def _correct_and_verify(
    variant_path: Path,
    backend: Backend,
    model: str,
    fix_compile_prompt: str,
    fix_exec_prompt: str,
    api_key: str,
    llm_cfg: LLMConfig,
    extra_flags: list[str] | None,
    profile: "ParadigmProfile | None",
    messages: list[dict],
    *,
    compile_variant: str,
    stage_label: str,
    agent_idx: int,
    serial_stdout: str,
    serial_checksum: float | None,
    serial_checksum_text: str | None,
    mpi_backend: Backend | None,
    hardware: "HardwareInfo | None",
    mpi_launch: "MpiLaunch | None",
    rtol: float,
) -> tuple[bool, str, list[tuple[str, ChatCompletion]]]:
    """Compile and verify, sharing one correction budget for both.

    Returns (ready, final_code, llm_results). ``ready`` means the variant
    compiled and matched the reference.
    """
    llm_results: list[tuple[str, ChatCompletion]] = []
    previous_error: str | None = None
    fixes_used = 0

    while True:
        ok, err = compile_file(
            variant_path, backend,
            variant=compile_variant,
            extra_flags=extra_flags,
            profile=profile,
        )
        if not ok:
            if previous_error is not None and err.strip() == previous_error.strip():
                logger.info(
                    "  [MoL] Agent %d: the error repeats; stopping fixes.",
                    agent_idx + 1,
                )
                _discard(agent_idx)
                return False, "", llm_results
            previous_error = err
            if fixes_used >= MOL_FIX_ATTEMPTS:
                logger.info(
                    "  [MoL] Agent %d: compilation failed; discarded.",
                    agent_idx + 1,
                )
                return False, "", llm_results
            fixes_used += 1
            current = variant_path.read_text(encoding="utf-8")
            label = (
                f"{stage_label} — MoL Agent {agent_idx + 1} "
                f"fix {fixes_used} (compilation)"
            )
            sent = _request_fix(
                messages=messages,
                content=f"{fix_compile_prompt}\n\n{current}\n\nCompilation errors:\n{err}",
                api_key=api_key, model=model, llm_cfg=llm_cfg, label=label,
                llm_results=llm_results, variant_path=variant_path,
                agent_idx=agent_idx, fix_number=fixes_used,
            )
            if not sent:
                _discard(agent_idx)
                return False, "", llm_results
            continue

        passed, reason = _verify(
            variant_path, backend, serial_stdout, serial_checksum,
            serial_checksum_text=serial_checksum_text,
            profile=profile, mpi_backend=mpi_backend,
            hardware=hardware, mpi_launch=mpi_launch,
            compile_variant=compile_variant, rtol=rtol,
        )
        if passed:
            return True, variant_path.read_text(encoding="utf-8"), llm_results

        logger.info(
            "  [MoL] Agent %d: incorrect result (%s).",
            agent_idx + 1, _one_line(reason),
        )
        if previous_error is not None and reason.strip() == previous_error.strip():
            logger.info(
                "  [MoL] Agent %d: the error repeats; stopping fixes.",
                agent_idx + 1,
            )
            _discard(agent_idx)
            return False, "", llm_results
        previous_error = reason
        if fixes_used >= MOL_FIX_ATTEMPTS or not fix_exec_prompt.strip():
            if not fix_exec_prompt.strip():
                logger.warning(
                    "  [MoL] Agent %d: fix_execution_errors.txt is missing.",
                    agent_idx + 1,
                )
            _discard(agent_idx)
            return False, "", llm_results
        fixes_used += 1
        current = variant_path.read_text(encoding="utf-8")
        label = (
            f"{stage_label} — MoL Agent {agent_idx + 1} "
            f"fix {fixes_used} (execution)"
        )
        sent = _request_fix(
            messages=messages,
            content=(
                f"{fix_exec_prompt}\n\n{current}\n\n"
                f"**Execution error details:**\n{reason}"
            ),
            api_key=api_key, model=model, llm_cfg=llm_cfg, label=label,
            llm_results=llm_results, variant_path=variant_path,
            agent_idx=agent_idx, fix_number=fixes_used,
        )
        if not sent:
            _discard(agent_idx)
            return False, "", llm_results


def _failure_reason(exc: ExecutionError) -> str:
    code = exc.returncode
    if code is None:
        head = "Execution failed"
    elif code < 0:
        head = f"Execution failed: signal {-code}"
    else:
        head = f"Execution failed: exit code {code}"
    tail = (exc.stderr or "").splitlines()[-30:]
    if tail:
        return head + "\n" + "\n".join(tail)
    return head


def _verify(
    variant_path: Path,
    backend: Backend,
    serial_stdout: str,
    serial_checksum: float | None,
    *,
    serial_checksum_text: str | None = None,
    profile: "ParadigmProfile | None",
    mpi_backend: Backend | None,
    hardware: "HardwareInfo | None",
    mpi_launch: "MpiLaunch | None",
    compile_variant: str,
    rtol: float = CHECKSUM_RTOL,
) -> tuple[bool, str]:
    """Run the variant once (configuration A) and compare it to the reference."""
    exe = executable_path(variant_path)
    cmd, env = effective_run(
        exe, backend,
        profile=profile, mpi_backend=mpi_backend,
        hardware=hardware,
        mpi_launch=mpi_launch,
        compile_variant=compile_variant,
    )
    try:
        stdout, _stderr, _elapsed = run_once(exe, cmd, env=env or None)
    except ExecutionTimeoutError:
        return False, f"Timeout: {EXECUTION_TIMEOUT_SECONDS} s"
    except ExecutionError as exc:
        return False, _failure_reason(exc)

    actual, actual_text = parse_checksum(stdout)
    if serial_checksum is not None and actual is None:
        return False, "No 'Checksum:' line in stdout"
    if serial_checksum is not None and actual is not None:
        ok, delta = checksums_match(
            serial_checksum, actual, rtol, serial_checksum_text,
        )
        if ok:
            return True, ""
        window = absolute_tolerance(serial_checksum, rtol, serial_checksum_text)
        expected = serial_checksum_text if serial_checksum_text else f"{serial_checksum}"
        got = actual_text if actual_text else f"{actual}"
        return False, (
            f"Checksum mismatch: expected {expected}, got {got}, "
            f"absolute difference {delta:.6g} (tolerance {window:.6g})."
        )
    if serial_stdout and stdout != serial_stdout:
        return False, first_stdout_diff(serial_stdout, stdout)
    return True, ""


def _bench_variant(
    variant_path: Path,
    backend: Backend,
    *,
    profile: "ParadigmProfile | None",
    mpi_backend: Backend | None,
    hardware: "HardwareInfo | None",
    mpi_launch: "MpiLaunch | None",
    compile_variant: str,
    stage_label: str,
    agent_index: int,
    logical_name: str = "",
) -> tuple[float, float | None, str | None, str]:
    """Warm up, measure, and return (wall_ms, kernel_ms, config, launch line)."""
    measured = benchmark_stage(
        variant_path, backend,
        profile=profile, mpi_backend=mpi_backend,
        hardware=hardware, mpi_launch=mpi_launch,
        compile_variant=compile_variant,
        logical_name=logical_name or f"{stage_label}-v{agent_index}",
    )
    return measured.wall_ms, measured.reported_ms, measured.thread_config, measured.launch_line


def _benchmark_input(
    input_path: Path,
    backend: Backend,
    *,
    compile_variant: str,
    profile: "ParadigmProfile | None",
    mpi_backend: Backend | None,
    hardware: "HardwareInfo | None",
    mpi_launch: "MpiLaunch | None",
    extra_flags: list[str] | None,
    stage_label: str,
    logical_name: str = "",
) -> tuple[float | None, float | None]:
    """Compile the input and benchmark it. Returns (wall_ms, reported_ms)."""
    logger.info("\n  [MoL] Compiling input code for the baseline…")
    ok, _err = compile_file(
        input_path, backend,
        variant=compile_variant,
        extra_flags=extra_flags,
        profile=profile,
    )
    if not ok:
        logger.warning("  [MoL] Could not compile the input for the baseline.")
        return None, None

    try:
        measured = benchmark_stage(
            input_path, backend,
            profile=profile, mpi_backend=mpi_backend,
            hardware=hardware, mpi_launch=mpi_launch,
            compile_variant=compile_variant,
            logical_name=logical_name or stage_label,
        )
        return measured.wall_ms, measured.reported_ms
    except Exception as exc:
        logger.warning("  [MoL] Baseline measurement failed: %s", exc)
        return None, None
    finally:
        # Don't remove the input binary; it may be reused by the next call to
        # benchmark_stage in pipeline_runner.
        pass


def _recompile_output(
    output_path: Path,
    backend: Backend,
    compile_variant: str,
    extra_flags: list[str] | None,
    profile: "ParadigmProfile | None",
) -> None:
    """Compile *output_path* so the binary is ready for the next pipeline stage.

    Called after writing the final code (winner or fallback) to *output_path*.
    Compilation failure is logged as a warning but does not raise — the pipeline
    will detect the missing binary when it calls benchmark_stage.
    """
    ok, err = compile_file(
        output_path, backend,
        variant=compile_variant,
        extra_flags=extra_flags,
        profile=profile,
    )
    if not ok:
        logger.warning(
            "  [MoL] Failed to recompile output_path:\n%s", err,
        )


_VERDICT_MARKDOWN = frozenset("*#`>-_")


def _first_verdict_line(text: str) -> str:
    """First non-empty line with leading markdown removed, original case."""
    first = ""
    for line in (text or "").splitlines():
        if line.strip():
            first = line.strip()
            break
    while first and first[0] in _VERDICT_MARKDOWN:
        first = first[1:].lstrip()
    return first


def _verdict_line(text: str) -> str:
    """First non-empty line, without leading markdown, in upper case."""
    return _first_verdict_line(text).upper()


def parse_audit_verdict(text: str) -> bool:
    """True only when the first non-empty line starts with LEGITIMATE.

    Leading spaces and markdown characters (``* # ` > - _``) are removed first.
    A line that starts with CHEAT is a rejection even if LEGITIMATE appears later.
    """
    first = _verdict_line(text)
    if not first or first.startswith("CHEAT"):
        return False
    return first.startswith("LEGITIMATE")


def _audit(
    *,
    original_code: str,
    optimized_code: str,
    audit_model: str,
    api_key: str,
    llm_cfg: LLMConfig,
    stage_label: str,
    rtol: float = CHECKSUM_RTOL,
    mode: Literal["optimization", "translation"] = "optimization",
) -> tuple[bool, str, ChatCompletion | None]:
    """Ask an auditor LLM whether the change is legitimate.

    Returns (passed, verdict line, ChatCompletion | None).

    Uses the raw message text (not ``extract_content``) because the auditor
    responds with a plain sentence, not source code.
    """
    if mode == "translation":
        opening = (
            "You are an expert code auditor for HPC optimization.\n"
            "This is a TRANSLATION stage: the OPTIMIZED code adds a new parallel "
            "paradigm (SIMD intrinsics, OpenMP threads or MPI processes) to the "
            "ORIGINAL code. Distributing work across threads or ranks, replacing "
            "the timer API, adding MPI_Init/MPI_Finalize, reductions of partial "
            "results and rank-0-only printing are expected and LEGITIMATE. Every "
            "element of the original timed kernel must still be computed exactly "
            "once over all ranks/threads.\n"
            "Compare the ORIGINAL code and the OPTIMIZED code below and decide whether "
            "the translation is LEGITIMATE or CHEATS.\n\n"
        )
        label = f"{stage_label} — Translation auditor"
    else:
        opening = (
            "You are an expert code auditor for HPC optimization.\n"
            "Compare the ORIGINAL code and the OPTIMIZED code below and decide whether "
            "the optimization is LEGITIMATE or CHEATS.\n\n"
        )
        label = f"{stage_label} — MoL Auditor"
    audit_prompt = (
        opening
        + "The harness has already verified that the OPTIMIZED program prints the same "
        "checksum as the serial reference within a relative tolerance of "
        f"{format_tolerance(rtol)}. Your job is to catch work that is skipped, moved "
        "or faked — not floating-point rounding differences.\n\n"
        "## CHEATS — reject the submission if any of these are true\n"
        "- It reduces the TOTAL work of the timed kernel: fewer elements updated, fewer "
        "arithmetic operations per element, fewer outer/time-step/convergence iterations, "
        "or skipped elements. Loop trip counts may legitimately change with tiling, "
        "blocking, vector strides or loop collapsing; the total work may not.\n"
        "- It moves computation out of the timed region (into initialization, before the "
        "start timer or after the stop timer), or reuses results across the timed repetitions.\n"
        "- It lowers the floating-point type (double → float), replaces a function with a "
        "lower-accuracy approximation that is not in the original, or skips convergence checks.\n"
        "- It eliminates work that affects the result, even if the checksum still matches "
        "within tolerance.\n"
        "- It changes the mathematical method (a different algorithm or formula).\n"
        "- It special-cases the input data or the checksum (hardcoded results, detecting the "
        "benchmark, computing only what the checksum reads).\n\n"
        "## NOT a cheat\n"
        "- Reassociating a sum or reduction (multiple independent accumulators, SIMD "
        "horizontal sums, OpenMP or MPI reductions, blocked accumulation over a reduction "
        "axis). The last bits of the result may differ; that is not a loss of precision.\n"
        "- Manual unroll or software prefetch that is redundant under -O3: not required, "
        "but not a cheat.\n\n"
        "## LEGITIMATE — accept the submission if every change belongs to these families\n"
        "- Tiling / loop interchange / loop fusion or fission: reorders or blocks loops for "
        "locality; same total work.\n"
        "- Register blocking / unroll-and-jam: computes a small block of outputs (several "
        "rows × several vectors) at once with independent accumulators kept in registers, "
        "reusing each loaded operand across the block. Inner loop strides change "
        "(e.g. i += 6, j += 8); every output element is still reduced over the same axis "
        "with the same operations.\n"
        "- Packing: copies blocks of the input arrays into temporary contiguous, aligned "
        "buffers (inside or outside the timed region, as in the original) so the inner "
        "kernel reads unit-stride data. The program's own arrays keep their layout.\n"
        "- Pack in-register (register-blocked tile transpose): loads a rectangular tile into "
        "vector registers, transposes it with shuffle/unpack/permute intrinsics and stores "
        "contiguously instead of gather/scatter.\n"
        "- SoA layout, `__restrict__`, `alignas` / `posix_memalign`, masked tail handling, "
        "FMA, aligned or non-temporal stores, prefetching.\n"
        "- Re-rolling a manual unroll into a single SIMD loop; introducing SIMD where a skip "
        "comment cited `log`/`exp`/`sqrt`/math functions; inlining a pure helper "
        "(Horner/minimax polynomial, `fabs`/`copysign`, per-lane blend) into SIMD while "
        "spilling only libm calls.\n"
        "- Thread-level tuning: `collapse`, `schedule` changes, `nowait`, NUMA first-touch "
        "initialization in parallel, padding against false sharing, thread-private "
        "accumulators or histograms combined with a reduction.\n"
        "- MPI: Alltoall or halo exchange instead of rank-0 Scatter+Gather, overlap of "
        "communication and computation, in-place collectives, persistent buffers.\n"
        "- A leading `/* ANALYSIS: ... */` comment block: documentation only.\n\n"
        "Respond with EXACTLY one line. The line must START with the word LEGITIMATE or "
        "the word CHEATS, followed by \" — \" and a brief reason.\n"
        "Example: LEGITIMATE — 6x8 register blocking with packed panels; same operations "
        "per output element\n"
        "Example: LEGITIMATE — pack in-register tile transpose replaces gather without "
        "changing the result\n"
        "Example: CHEATS — outer loop iterations reduced from N to N/2\n\n"
        f"## ORIGINAL CODE\n\n```c\n{original_code}\n```\n\n"
        f"## OPTIMIZED CODE\n\n```c\n{optimized_code}\n```\n"
    )

    try:
        data = chat_completion(
            api_key,
            [{"role": "user", "content": audit_prompt}],
            label=label,
            model=audit_model,
            api_url=llm_cfg.api_url,
            model_provider=llm_cfg.model_provider,
            thinking=False,
            thinking_effort=None,
        )
    except Exception as exc:
        logger.warning("  [MoL] Auditor call failed: %s", exc)
        return False, "", None

    # Read raw text — not extract_content, which expects source code.
    message = extract_message(data)
    raw = (message.content or "").strip()
    if not raw and message.reasoning:
        raw = message.reasoning.strip()

    passed = parse_audit_verdict(raw)
    if not passed and not _verdict_line(raw).startswith("CHEAT"):
        logger.warning(
            "  [MoL] Ambiguous auditor response (%r). Rejecting as a precaution.",
            raw[:120],
        )
    return passed, _first_verdict_line(raw), data
