"""Orchestrates multi-stage translation pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

from translator.config.settings import API_URL, BENCHMARK_RUNS
from translator.domain.pipeline import Pipeline
from translator.exceptions import (
    MaxRetriesExhaustedError,
    MpiHardwareError,
    PipelineStageError,
)
from translator.infrastructure.compilation import compile_file, executable_path
from translator.infrastructure.execution.benchmark import (
    StageTimingResult,
    benchmark,
    benchmark_stage,
    verify_and_benchmark,
)
from translator.services.metrics import MetricsAccumulator, print_metrics
from translator.services.stage_runner import LLMConfig, run_translation_stage

logger = logging.getLogger(__name__)

_SEP = "═" * 50
_SEP2 = "-" * 50


# ── Helpers de presentación ───────────────────────────────────────────────────

def _fmt_ms(value: float | None) -> str:
    """Format a millisecond value for display, or 'N/A'."""
    if value is None:
        return "N/A"
    return f"{value:.1f} ms"


def _fmt_speedup(num: float | None, denom: float | None) -> str:
    """Compute and format speedup = num / denom, or 'N/A'."""
    if num is None or denom is None or denom == 0.0:
        return "N/A"
    return f"{num / denom:.2f}x"


_ND = "n.d."
_FINAL_MPI_STAGES = frozenset({"mpi", "optmpi"})


def final_mpi_thread_config(stage_results: list[StageTimingResult]) -> str | None:
    """Binding name for the final MPI binary.

    An accepted OptMPI stage replaces MPI. A rejected OptMPI stage leaves
    the configuration chosen for MPI. Returns ``"A"``, ``"B"``, or None.
    """
    chosen: str | None = None
    for result in stage_results:
        name = result.stage_name.lower()
        if name == "mpi" and result.thread_config:
            chosen = result.thread_config
        elif name == "optmpi" and result.accepted and result.thread_config:
            chosen = result.thread_config
    return chosen


def binding_named(hardware, name: str | None, *, per_rank: bool):
    """Return the A/B placement with this name, or None when it does not exist."""
    if not name:
        return None
    from translator.infrastructure.hardware import omp_bindings

    for binding in omp_bindings(hardware, per_rank=per_rank):
        if binding.name == name:
            return binding
    return None


def _log_stage_timings(
    stage_result: StageTimingResult,
    serial_wall_ms: float,
    serial_reported_ms: float | None,
    prev_result: StageTimingResult | None,
    runs: int,
) -> None:
    """Log timing summary immediately after a pipeline stage completes."""
    logger.info("\n%s", _SEP2)
    logger.info(
        "  Execution times — %s (median of %s runs)",
        stage_result.stage_name, runs,
    )
    logger.info("%s", _SEP2)

    if not stage_result.accepted:
        logger.info("  Kernel:             %s", _ND)
        logger.info("  Speedup vs serial:   %s", _ND)
        if prev_result is not None:
            logger.info(
                "  Speedup vs previous stage (%s):  %s",
                prev_result.stage_name,
                _ND,
            )
        logger.info("  Process time (startup, init and communication): %s", _ND)
        return

    if stage_result.thread_config:
        logger.info("  Configuration:       %s", stage_result.thread_config)

    if stage_result.reported_ms is not None:
        logger.info("  Kernel:             %s", _fmt_ms(stage_result.reported_ms))
        logger.info(
            "  Speedup vs serial:   %s",
            _fmt_speedup(serial_reported_ms, stage_result.reported_ms),
        )
        if prev_result is not None and prev_result.reported_ms is not None:
            logger.info(
                "  Speedup vs previous stage (%s):  %s",
                prev_result.stage_name,
                _fmt_speedup(prev_result.reported_ms, stage_result.reported_ms),
            )
        logger.info(
            "  Process time (startup, init and communication): %s",
            _fmt_ms(stage_result.wall_ms),
        )
    else:
        logger.info(
            "  Process time (startup, init and communication): %s",
            _fmt_ms(stage_result.wall_ms),
        )
        logger.info(
            "  Speedup vs serial:   %s",
            _fmt_speedup(serial_wall_ms, stage_result.wall_ms),
        )
        if prev_result is not None:
            logger.info(
                "  Speedup vs previous stage (%s):  %s",
                prev_result.stage_name,
                _fmt_speedup(prev_result.wall_ms, stage_result.wall_ms),
            )
        logger.info(
            "  Kernel:              N/A  "
            "(the program does not print 'Tiempo de ejecucion ... ms')"
        )


def _log_pipeline_evolution(
    stage_results: list[StageTimingResult],
    serial_wall_ms: float,
    serial_reported_ms: float | None,
    stage_names: list[str],
) -> None:
    """Print the full pipeline timing evolution after all stages finish."""
    sep = "═" * 50
    logger.info("\n\n%s", sep)
    logger.info("  Timing evolution — pipeline")
    logger.info("%s", sep)

    stage_labels = ["serial"] + stage_names
    all_reported = [serial_reported_ms] + [r.reported_ms for r in stage_results]
    has_kernel = serial_reported_ms is not None or any(
        r.accepted and r.reported_ms is not None for r in stage_results
    )

    def _shown_ms(result: StageTimingResult, value: float | None) -> str:
        if not result.accepted:
            return _ND
        return _fmt_ms(value)

    def _shown_speedup(
        result: StageTimingResult, num: float | None, denom: float | None,
    ) -> str:
        if not result.accepted:
            return _ND
        return _fmt_speedup(num, denom)

    if has_kernel:
        rep_chain = [_fmt_ms(serial_reported_ms)] + [
            _shown_ms(r, r.reported_ms) for r in stage_results
        ]
        logger.info(
            "\n  Kernel\n  %s",
            "  →  ".join(f"{lbl}: {t}" for lbl, t in zip(stage_labels, rep_chain)),
        )
        rep_speedups_serial = [
            _shown_speedup(r, serial_reported_ms, r.reported_ms) for r in stage_results
        ]
        logger.info(
            "  Speedup vs serial:  %s",
            "  →  ".join(f"{lbl}: {s}" for lbl, s in zip(stage_names, rep_speedups_serial)),
        )
        rep_speedups_prev = [
            _shown_speedup(r, all_reported[i], all_reported[i + 1])
            for i, r in enumerate(stage_results)
        ]
        logger.info(
            "  Speedup vs previous: %s",
            "  →  ".join(f"{lbl}: {s}" for lbl, s in zip(stage_names, rep_speedups_prev)),
        )

    wall_chain = [_fmt_ms(serial_wall_ms)] + [
        _shown_ms(r, r.wall_ms) for r in stage_results
    ]
    logger.info(
        "\n  Process time (startup, init and communication)\n  %s",
        "  →  ".join(f"{lbl}: {t}" for lbl, t in zip(stage_labels, wall_chain)),
    )
    if not has_kernel:
        wall_speedups_serial = [
            _shown_speedup(r, serial_wall_ms, r.wall_ms) for r in stage_results
        ]
        logger.info(
            "  Speedup vs serial:  %s",
            "  →  ".join(f"{lbl}: {s}" for lbl, s in zip(stage_names, wall_speedups_serial)),
        )
        all_wall = [serial_wall_ms] + [r.wall_ms for r in stage_results]
        wall_speedups_prev = [
            _shown_speedup(r, all_wall[i], all_wall[i + 1])
            for i, r in enumerate(stage_results)
        ]
        logger.info(
            "  Speedup vs previous: %s",
            "  →  ".join(f"{lbl}: {s}" for lbl, s in zip(stage_names, wall_speedups_prev)),
        )
        logger.info("\n  Kernel: N/A for every stage")

    logger.info("\n  Resources and speedup per node")
    for result in stage_results:
        if result.accepted and result.nodes > 0:
            if serial_reported_ms is not None and result.reported_ms:
                speedup = serial_reported_ms / result.reported_ms
            elif result.wall_ms:
                speedup = serial_wall_ms / result.wall_ms
            else:
                speedup = None
            per_node = f"{speedup / result.nodes:.2f}x" if speedup is not None else _ND
            threads = result.thread_config or "—"
        else:
            per_node = _ND
            threads = _ND
        logger.info(
            "  %s: nodes %s, physical cores %s, threads %s, speedup per node %s",
            result.stage_name,
            result.nodes if result.accepted else _ND,
            result.physical_cores if result.accepted else _ND,
            threads,
            per_node,
        )

    logger.info("\n%s\n", sep)


def _vectorization_report(stage, input_path: Path, extra_flags: list[str] | None) -> str | None:
    """GCC loop report for the AVX translation and later Opt* stages."""
    wants_report = (not stage.is_optimization and stage.backend.slug == "avx") or (
        stage.is_optimization and stage.profile.avx
    )
    if not wants_report:
        return None
    from translator.infrastructure.compilation import translation_compile_argv
    from translator.infrastructure.compilation.vec_report import vec_report_block

    built = translation_compile_argv(
        input_path,
        stage.backend,
        variant=stage.compile_variant,
        extra_flags=extra_flags,
        profile=stage.profile,
    )
    if built is None:
        return None
    cmd, _label = built
    return vec_report_block(input_path, cmd)


# ── Orquestador principal ─────────────────────────────────────────────────────

def run_pipeline(
    source_path: Path,
    api_key: str,
    *,
    pipeline: Pipeline,
    model: str,
    api_url: str = API_URL,
    model_provider: str | None = None,
    thinking: bool = False,
    thinking_effort: str | None = None,
    extra_flags: list[str] | None = None,
    compare: bool = True,
    metrics: bool = False,
    mol_agent_models: list[str] | None = None,
    mol_audit_model: str | None = None,
    debug: bool = False,
    rtol: float | None = None,
) -> None:
    """Execute a multi-stage translation pipeline.

    For each stage:
    1. The output of stage N is the input of stage N+1.
    2. The LLM conversation is reset at the start of every stage.
    3. A single serial reference (stdout + checksum) is captured once at the
       start and reused for ``--compare`` verification and MoL correctness
       checks across all stages.
    4. Timing measurements are taken after every stage and shown immediately.
    5. A full pipeline evolution summary is printed at the end.
    6. Compilation and execution for every stage use the *accumulated* paradigm
       profile rather than per-stage hybrid flags, so MPI+OpenMP+AVX is handled
       correctly regardless of stage order.

    Args:
        source_path:       Original serial source file.
        api_key:           API key for the LLM provider.
        pipeline:          Resolved :class:`~translator.domain.pipeline.Pipeline`.
        model:             Model alias or ID for translation stages.
        api_url:           LLM API endpoint URL.
        model_provider:    OpenRouter provider override (``provider.only``).
        thinking:          Enable extended reasoning.
        thinking_effort:   Reasoning effort level.
        extra_flags:       Extra compiler flags forwarded to every stage.
        compare:           Verify execution correctness after each translation stage.
        metrics:           Accumulate and print LLM metrics at the end.
        mol_agent_models:  Model IDs for MoL generator agents (optimization stages).
        mol_audit_model:   Model ID for the MoL auditor.
    """
    n_stages = len(pipeline.stages)
    logger.info("\n\n%s", _SEP)
    logger.info("  Pipeline: %s  (%s stages)", pipeline.name, n_stages)
    for s in pipeline.stages:
        logger.info(
            "    Stage %s/%s: %s%s",
            s.stage_index + 1, n_stages, s.logical_name,
            " [MoL optimization]" if s.is_optimization else "",
        )
    logger.info("%s", _SEP)

    # ── Hardware info for launch and optimisation prompts ─────────────────
    from translator.infrastructure.hardware import capture_hardware_info
    hw_info = capture_hardware_info()
    if hw_info:
        logger.info("\n  Detected hardware:\n%s", hw_info.to_prompt_block())
    else:
        logger.info("\n  [WARNING] Could not obtain hardware information (lscpu unavailable).")

    from translator.infrastructure.hardware import (
        MPI_HARDWARE_MESSAGE,
        decide_mpi_launch,
        mpirun_command,
    )

    mpi_launch = None
    if any(stage.profile.mpi for stage in pipeline.stages):
        mpi_launch = decide_mpi_launch(hw_info)
        if not mpi_launch.supported:
            logger.error("\n  %s", MPI_HARDWARE_MESSAGE)
            raise MpiHardwareError(MPI_HARDWARE_MESSAGE)
        logger.info(
            "\n  MPI launch: %s ranks, %s OpenMP threads per rank, %s",
            mpi_launch.ranks,
            mpi_launch.omp_threads,
            mpi_launch.placement,
        )
        logger.info(
            "  %s",
            " ".join(mpirun_command("<programa>", mpi_launch)[:-1]),
        )

    llm_cfg = LLMConfig(
        model=model,
        api_url=api_url,
        model_provider=model_provider,
        thinking=thinking,
        thinking_effort=thinking_effort,
    )

    acc: MetricsAccumulator | None = MetricsAccumulator() if metrics else None

    # ── Find the MPI backend for hybrid execution ─────────────────────────
    # This backend provides the mpirun run_cmd / run_env for any stage whose
    # accumulated profile includes MPI.
    mpi_backend = next(
        (s.backend for s in pipeline.stages if s.backend.slug == "mpi"),
        None,
    )

    # ── Serial baseline (compilation + timing) ────────────────────────────
    first_backend = pipeline.stages[0].backend
    logger.info("\n\n  Compiling the serial version for the timing baseline…")
    from translator.exceptions import CompilationError
    ok_serial, err_serial = compile_file(
        source_path, first_backend, variant="serial", extra_flags=extra_flags,
    )
    if not ok_serial:
        raise CompilationError("Could not compile the serial version for the baseline:", output=err_serial)

    serial_exe = executable_path(source_path)
    serial_cmd = first_backend.run_serial_cmd(serial_exe)
    serial_launch = " ".join(serial_cmd)
    serial_stdout_raw, _s_stderr, serial_wall_avg, serial_reported_avg = benchmark(
        serial_exe, serial_cmd, BENCHMARK_RUNS, label="serial",
    )
    serial_wall_ms = serial_wall_avg * 1000.0
    serial_reported_ms = serial_reported_avg  # already ms or None

    # ── Serial reference for comparison and MoL ───────────────────────────
    # Always capture reference stdout/checksum (not just when --compare is
    # active) because MoL optimisation stages need it unconditionally.
    from translator.infrastructure.execution.output_parser import parse_checksum as _parse_checksum
    serial_checksum, serial_checksum_text = _parse_checksum(serial_stdout_raw)
    serial_stdout = serial_stdout_raw

    if serial_exe.is_file():
        serial_exe.unlink()

    if serial_checksum is not None:
        logger.info("  Serial reference checksum: %.6g", serial_checksum)
    else:
        logger.info("  (Checksum unavailable in the serial run — stdout will be compared exactly)")

    # ── MoL configuration ─────────────────────────────────────────────────
    from translator.config.settings import (
        CHECKSUM_RTOL,
        MOL_DEFAULT_AGENT_MODELS,
        MOL_DEFAULT_AUDIT_MODEL,
        MOL_SPEEDUP_AUDIT_THRESHOLD,
    )
    if rtol is None:
        rtol = CHECKSUM_RTOL
    from translator.services.mol_runner import MoLConfig

    mol_cfg = MoLConfig(
        agent_models=mol_agent_models or MOL_DEFAULT_AGENT_MODELS,
        audit_model=mol_audit_model or MOL_DEFAULT_AUDIT_MODEL,
        speedup_audit_threshold=MOL_SPEEDUP_AUDIT_THRESHOLD,
    )

    stage_results: list[StageTimingResult] = []
    # Use logical_name for stage labels in the evolution summary
    stage_names: list[str] = [s.logical_name for s in pipeline.stages]

    try:
        for stage in pipeline.stages:
            input_path = (
                source_path
                if stage.stage_index == 0
                else pipeline.artifact_path(source_path, stage.stage_index - 1)
            )
            output_path = pipeline.artifact_path(source_path, stage.stage_index)

            profile = stage.profile

            logger.info("\n%s", _SEP)
            logger.info(
                "  Stage %s/%s: %s  [profile: %s]",
                stage.stage_index + 1, n_stages, stage.logical_name,
                profile.label(),
            )
            logger.info("  Input:  %s", input_path.name)
            logger.info("  Output: %s", output_path.name)
            if stage.is_optimization:
                logger.info("  Mode:   MoL optimization")
            logger.info("%s", _SEP)

            stage_label = (
                f"Stage {stage.stage_index + 1}/{n_stages} {stage.logical_name}"
            )

            accepted = True
            mol_wall: float | None = None
            mol_reported: float | None = None
            mol_config: str | None = None
            mol_launch = ""
            try:
                vec_report = _vectorization_report(stage, input_path, extra_flags)
                if stage.is_optimization:
                    # ── Optimisation stage (MoL) ──────────────────────────
                    from translator.services.mol_runner import run_optimization_stage
                    from translator.infrastructure.hardware import include_mpi_launch_prompt
                    from translator.services.prompt_blocks import (
                        compose_stage_prompt,
                        performance_block,
                    )

                    if stage.stage_index == 0 or not stage_results:
                        base_wall, base_reported = serial_wall_ms, serial_reported_ms
                        base_launch = serial_launch
                    else:
                        previous = stage_results[-1]
                        base_wall = previous.wall_ms
                        base_reported = previous.reported_ms
                        base_launch = previous.launch_line

                    translate_prompt = compose_stage_prompt(
                        stage.prompts.translate.read_text(encoding="utf-8"),
                        hardware=hw_info,
                        mpi_launch=mpi_launch,
                        include_mpi_launch=include_mpi_launch_prompt(
                            stage.backend.slug, stage.compile_variant,
                        ),
                        logical_name=stage.logical_name,
                        profile_mpi=profile.mpi,
                        rtol=rtol,
                        performance=performance_block(
                            process_ms=base_wall,
                            kernel_ms=base_reported,
                            launch_line=base_launch,
                            serial_kernel_ms=serial_reported_ms,
                        ),
                        vec_report=vec_report,
                    )
                    fix_compile_prompt = stage.prompts.fix_compile.read_text(encoding="utf-8")
                    fix_exec_prompt = ""
                    if stage.prompts.fix_execution is not None:
                        fix_exec_prompt = stage.prompts.fix_execution.read_text(encoding="utf-8")

                    accepted, mol_wall, mol_reported, mol_config, mol_launch = run_optimization_stage(
                        input_path,
                        output_path,
                        backend=stage.backend,
                        compile_variant=stage.compile_variant,
                        translate_prompt=translate_prompt,
                        fix_compile_prompt=fix_compile_prompt,
                        fix_exec_prompt=fix_exec_prompt,
                        llm_cfg=llm_cfg,
                        mol_cfg=mol_cfg,
                        api_key=api_key,
                        extra_flags=extra_flags,
                        serial_stdout=serial_stdout,
                        serial_checksum=serial_checksum,
                        serial_checksum_text=serial_checksum_text,
                        profile=profile,
                        mpi_backend=mpi_backend,
                        hardware=hw_info,
                        mpi_launch=mpi_launch,
                        acc=acc,
                        stage_label=stage_label,
                        debug=debug,
                        baseline_wall_ms=base_wall,
                        baseline_reported_ms=base_reported,
                        logical_name=stage.logical_name,
                        rtol=rtol,
                    )
                else:
                    # ── Translation stage ─────────────────────────────────
                    run_translation_stage(
                        input_path,
                        output_path,
                        backend=stage.backend,
                        translate_prompt_path=stage.prompts.translate,
                        fix_compile_prompt_path=stage.prompts.fix_compile,
                        fix_exec_prompt_path=stage.prompts.fix_execution,
                        llm_cfg=llm_cfg,
                        api_key=api_key,
                        extra_flags=extra_flags,
                        compare=compare,
                        serial_stdout=serial_stdout,
                        serial_checksum=serial_checksum,
                        serial_checksum_text=serial_checksum_text,
                        profile=profile,
                        mpi_backend=mpi_backend,
                        hardware=hw_info,
                        mpi_launch=mpi_launch,
                        compile_variant=stage.compile_variant,
                        acc=acc,
                        stage_label=stage_label,
                        logical_name=stage.logical_name,
                        rtol=rtol,
                        vec_report=vec_report,
                        audit_model=mol_cfg.audit_model,
                    )
            except MaxRetriesExhaustedError as exc:
                raise PipelineStageError(
                    str(exc),
                    stage_name=stage.logical_name,
                    stage_index=stage.stage_index,
                ) from exc

            # ── Measure this stage's timing ──────────────────────────────
            from translator.infrastructure.hardware import stage_resources

            if accepted and stage.is_optimization and mol_wall is not None:
                nodes, cores = stage_resources(
                    profile, hw_info, mpi_launch, stage.compile_variant,
                )
                stage_result = StageTimingResult(
                    stage_name=stage.logical_name,
                    wall_ms=mol_wall,
                    reported_ms=mol_reported,
                    thread_config=mol_config,
                    nodes=nodes,
                    physical_cores=cores,
                    launch_line=mol_launch,
                )
            elif accepted:
                stage_result = benchmark_stage(
                    output_path, stage.backend,
                    profile=profile,
                    mpi_backend=mpi_backend,
                    hardware=hw_info,
                    mpi_launch=mpi_launch,
                    compile_variant=stage.compile_variant,
                    logical_name=stage.logical_name,
                    runs=BENCHMARK_RUNS,
                )
            else:
                if stage_results:
                    carried = stage_results[-1]
                    carried_wall = carried.wall_ms
                    carried_reported = carried.reported_ms
                    carried_launch = carried.launch_line
                    carried_nodes = carried.nodes
                    carried_cores = carried.physical_cores
                else:
                    carried_wall, carried_reported = serial_wall_ms, serial_reported_ms
                    carried_launch = serial_launch
                    carried_nodes, carried_cores = 1, 1
                stage_result = StageTimingResult(
                    stage_name=stage.logical_name,
                    wall_ms=carried_wall,
                    reported_ms=carried_reported,
                    accepted=False,
                    nodes=carried_nodes,
                    physical_cores=carried_cores,
                    launch_line=carried_launch,
                )
            stage_results.append(stage_result)

            prev_result = stage_results[-2] if len(stage_results) >= 2 else None
            _log_stage_timings(
                stage_result,
                serial_wall_ms,
                serial_reported_ms,
                prev_result,
                BENCHMARK_RUNS,
            )

            if stage.is_last:
                # Full verify_and_benchmark for the final stage.
                # MPI and OptMPI keep the A/B placement that won the timing.
                # An accepted OptMPI overrides MPI; a rejected one keeps MPI's.
                final_binding = None
                if stage.logical_name.lower() in _FINAL_MPI_STAGES:
                    final_binding = binding_named(
                        hw_info,
                        final_mpi_thread_config(stage_results),
                        per_rank=True,
                    )
                verify_and_benchmark(
                    source_path,
                    stage.backend,
                    extra_flags=extra_flags,
                    translated_path=output_path,
                    profile=profile,
                    mpi_backend=mpi_backend,
                    hardware=hw_info,
                    mpi_launch=mpi_launch,
                    compile_variant=stage.compile_variant,
                    rtol=rtol,
                    binding=final_binding,
                )
            else:
                logger.info(
                    "\n\n  Stage %s/%s (%s) finished. Continuing with the next stage…",
                    stage.stage_index + 1, n_stages, stage.logical_name,
                )

        # ── Full pipeline evolution ────────────────────────────────────────
        if len(stage_results) > 1:
            _log_pipeline_evolution(
                stage_results,
                serial_wall_ms,
                serial_reported_ms,
                stage_names,
            )

    finally:
        if acc is not None and acc.calls:
            acc.resolve_generation_metadata(api_key)
            print_metrics(acc)
