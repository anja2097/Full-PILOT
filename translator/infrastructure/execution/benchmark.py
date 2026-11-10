"""Benchmarking and correctness verification: serial vs translated binary."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from translator.config.settings import BENCHMARK_RUNS, CHECKSUM_RTOL, EXECUTION_TIMEOUT_SECONDS
from translator.domain.backend import Backend
from translator.infrastructure.execution.output_parser import (
    absolute_tolerance,
    checksums_match,
    parse_checksum,
    parse_stdout_time,
)
from translator.infrastructure.hardware import (
    HardwareInfo,
    MpiLaunch,
    OmpBinding,
    decide_mpi_launch,
    mpirun_command,
    omp_bindings,
    stage_resources,
)
from translator.exceptions import CompilationError, ExecutionError, ExecutionTimeoutError
from translator.infrastructure.compilation import compile_file, executable_path
from translator.infrastructure.execution.process import run_once

if TYPE_CHECKING:
    from translator.domain.pipeline import ParadigmProfile

logger = logging.getLogger(__name__)


@dataclass
class StageTimingResult:
    """Timing measurements for one pipeline stage."""

    stage_name: str
    wall_ms: float              # median wall-clock time in ms (measured from Python)
    reported_ms: float | None   # median self-reported time in ms (from program stdout), or None
    accepted: bool = True       # False when an Opt* stage kept the previous code
    thread_config: str | None = None   # "A", "B", or None when only one launch exists
    nodes: int = 1
    physical_cores: int = 1
    launch_line: str = ""


DUAL_MEASURE_STAGES = frozenset({"openmp", "optopenmp", "mpi", "optmpi"})


def median(values: list[float]) -> float:
    """Median of *values*. Even counts use the mean of the two central values."""
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _format_execute_cmd(cmd: list[str], env: dict[str, str] | None = None) -> str:
    parts = [f"{key}={value}" for key, value in (env or {}).items()]
    parts.extend(cmd)
    return " ".join(parts)


def _log_execute_header(
    label: str,
    cmd: list[str],
    runs: int,
    *,
    env: dict[str, str] | None = None,
) -> None:
    logger.info("\n\n  Running (%s): %s", label, _format_execute_cmd(cmd, env))
    logger.info("  Per-run limit: %s s", EXECUTION_TIMEOUT_SECONDS)
    if runs > 1:
        logger.info("  Repetitions: %s", runs)


def _cleanup_executables(*exe_paths: Path) -> None:
    removed: list[Path] = []
    for exe_path in exe_paths:
        if exe_path.is_file():
            exe_path.unlink()
            removed.append(exe_path)
    if removed:
        logger.info("\n  Executables removed:")
        for exe_path in removed:
            logger.info("    %s", exe_path)


def _apply_binding(env: dict[str, str], binding: OmpBinding) -> dict[str, str]:
    updated = dict(env)
    updated["OMP_NUM_THREADS"] = str(binding.threads)
    updated["OMP_PLACES"] = binding.places
    updated["OMP_PROC_BIND"] = binding.proc_bind
    return updated


def effective_run(
    exe: Path,
    backend: Backend,
    *,
    profile: "ParadigmProfile | None",
    mpi_backend: Backend | None,
    hardware: HardwareInfo | None = None,
    mpi_launch: MpiLaunch | None = None,
    compile_variant: str = "translated",
    binding: OmpBinding | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return the (cmd, env) pair to execute *exe* based on accumulated profile.

    Logic:
    - ``compile_variant == "serial"`` → always run without mpirun, regardless
      of which backend the stage borrowed.
    - ``profile.mpi`` is True → the Open MPI 4.1.1 command from *mpi_launch*,
      or from *hardware* and Slurm when the caller did not pass one.
      *binding* selects configuration A or B; the default is A.
    - ``profile.openmp`` without MPI → the backend command, with OpenMP
      placement from *binding* or configuration A.
    - Otherwise → run via ``backend.run_cmd`` (no mpirun prefix).
    """
    if compile_variant == "serial":
        return backend.run_serial_cmd(exe), {}
    if profile is not None and profile.mpi and mpi_backend is not None:
        launch = mpi_launch if mpi_launch is not None else decide_mpi_launch(hardware)
        return mpirun_command(str(exe), launch, binding), mpi_backend.run_env()
    if profile is not None and profile.openmp:
        chosen = binding or omp_bindings(hardware, per_rank=False)[0]
        return backend.run_cmd(exe), _apply_binding(backend.run_env(), chosen)
    return backend.run_cmd(exe), backend.run_env()


def candidate_launches(
    exe: Path,
    backend: Backend,
    *,
    profile: "ParadigmProfile | None",
    mpi_backend: Backend | None,
    hardware: HardwareInfo | None,
    mpi_launch: MpiLaunch | None,
    compile_variant: str,
    logical_name: str,
) -> list[tuple[str | None, list[str], dict[str, str]]]:
    """Launches to measure. OpenMP and MPI stages try A and B; others try one.

    Each item is ``(config name or None, command, env)``.
    """
    dual = (
        compile_variant != "serial"
        and logical_name.lower() in DUAL_MEASURE_STAGES
        and profile is not None
        and (profile.mpi or profile.openmp)
    )
    if not dual:
        cmd, env = effective_run(
            exe, backend,
            profile=profile, mpi_backend=mpi_backend, hardware=hardware,
            mpi_launch=mpi_launch, compile_variant=compile_variant,
        )
        return [(None, cmd, env)]

    per_rank = bool(profile is not None and profile.mpi)
    bindings = omp_bindings(hardware, per_rank=per_rank)
    launches: list[tuple[str | None, list[str], dict[str, str]]] = []
    for binding in bindings:
        cmd, env = effective_run(
            exe, backend,
            profile=profile, mpi_backend=mpi_backend, hardware=hardware,
            mpi_launch=mpi_launch, compile_variant=compile_variant,
            binding=binding,
        )
        launches.append((binding.name, cmd, env))
    return launches


def benchmark(
    exe_path: Path,
    cmd: list[str],
    runs: int,
    *,
    label: str,
    env: dict[str, str] | None = None,
) -> tuple[str, str, float, float | None]:
    """Warm up once, then return the median of ``runs`` measured executions.

    Returns ``(stdout, stderr, wall_median_s, kernel_median_ms)``.

    - wall_median_s: median wall-clock time measured from Python (seconds).
    - kernel_median_ms: median internal time reported by the program (ms),
      or None if any measured run does not print 'Tiempo de ejecucion'.
    stdout and stderr come from the last measured run.
    """
    _log_execute_header(label, cmd, runs, env=env)
    logger.info("  Warmup: 1 unmeasured run")
    run_once(exe_path, cmd, env=env)
    stdout, stderr = "", ""
    walls: list[float] = []
    kernels: list[float] = []
    kernel_available = True
    for _ in range(runs):
        stdout, stderr, elapsed = run_once(exe_path, cmd, env=env)
        walls.append(elapsed)
        parsed = parse_stdout_time(stdout)
        if parsed is None:
            kernel_available = False
        else:
            kernels.append(parsed)
    logger.info("  Result: execution OK")
    kernel_median = median(kernels) if kernel_available and kernels else None
    return stdout, stderr, median(walls), kernel_median


def _format_sample_time(wall_s: float, kernel_ms: float | None) -> str:
    process = f"proceso {wall_s * 1000.0:.1f} ms"
    if kernel_ms is None:
        return process
    return f"kernel {kernel_ms:.1f} ms, {process}"


def benchmark_stage(
    artifact_path: Path,
    backend: Backend,
    *,
    profile: "ParadigmProfile | None" = None,
    mpi_backend: Backend | None = None,
    hardware: HardwareInfo | None = None,
    mpi_launch: MpiLaunch | None = None,
    compile_variant: str = "translated",
    logical_name: str = "",
    runs: int = BENCHMARK_RUNS,
) -> StageTimingResult:
    """Compile-free benchmark of an already-compiled pipeline stage artifact.

    One warm-up plus *runs* measured executions. OpenMP and MPI stages measure
    configurations A and B and keep the faster. The binary must already exist.
    """
    exe = executable_path(artifact_path)
    launches = candidate_launches(
        exe, backend,
        profile=profile,
        mpi_backend=mpi_backend,
        hardware=hardware,
        mpi_launch=mpi_launch,
        compile_variant=compile_variant,
        logical_name=logical_name,
    )
    samples: list[tuple[str | None, list[str], dict[str, str], float, float | None]] = []
    for name, cmd, env in launches:
        _stdout, _stderr, wall_s, kernel_ms = benchmark(
            exe, cmd, runs,
            label=f"{logical_name or backend.name} {name or ''}".strip(),
            env=env or None,
        )
        samples.append((name, cmd, env, wall_s, kernel_ms))
        if name:
            logger.info("  Median %s: %s", name, _format_sample_time(wall_s, kernel_ms))

    use_kernel = all(sample[4] is not None for sample in samples)

    def _metric(sample: tuple[str | None, list[str], dict[str, str], float, float | None]) -> float:
        if use_kernel and sample[4] is not None:
            return sample[4]
        return sample[3]

    winner = min(samples, key=_metric)
    if any(sample[0] for sample in samples):
        chosen = winner[0] or "only"
        if use_kernel and winner[4] is not None:
            logger.info(
                "  Chosen configuration: %s (kernel %.1f ms)",
                chosen, winner[4],
            )
        else:
            logger.info(
                "  Chosen configuration: %s (process %.1f ms)",
                chosen, winner[3] * 1000.0,
            )
    nodes, cores = stage_resources(profile, hardware, mpi_launch, compile_variant)
    return StageTimingResult(
        stage_name=logical_name or backend.name,
        wall_ms=winner[3] * 1000.0,
        reported_ms=winner[4],
        thread_config=winner[0],
        nodes=nodes,
        physical_cores=cores,
        launch_line=_format_execute_cmd(winner[1], winner[2] or None),
    )


def run_and_check(
    original_path: Path,
    backend: Backend,
    *,
    serial_stdout: str,
    serial_checksum: float | None,
    extra_flags: list[str] | None = None,
    translated_path: Path | None = None,
    profile: "ParadigmProfile | None" = None,
    mpi_backend: Backend | None = None,
    hardware: HardwareInfo | None = None,
    mpi_launch: MpiLaunch | None = None,
    compile_variant: str = "translated",
    rtol: float = CHECKSUM_RTOL,
    serial_checksum_text: str | None = None,
) -> tuple[bool, str]:
    """Execute the translated binary once and compare its output against the serial reference.

    This function is designed for the ``--compare`` feedback loop: it runs the translated
    binary a single time (not benchmarking), captures any runtime failure, and returns a
    structured error description suitable for feeding back to the LLM.

    Args:
        original_path:  Path to the original serial source file (used to derive the
                        translated path when *translated_path* is not supplied).
        backend:        The active backend.
        serial_stdout:  Captured stdout of the serial reference run.
        serial_checksum: Pre-parsed checksum from the serial run (or None if absent).
        extra_flags:    Extra compilation/run flags.
        translated_path: Explicit path to the translated source/binary.  When
                         ``None``, the path is derived via ``backend.translated_path``.
        profile:        Accumulated paradigm profile — determines the run command.
        mpi_backend:    MPI backend for execution when ``profile.mpi`` is True.
        compile_variant: ``"serial"`` → run without mpirun even if profile has MPI.

    Returns:
        ``(True, "")`` if the binary ran successfully and its output is within
        tolerance of the serial reference.
        ``(False, error_description)`` otherwise.
    """
    t_path = translated_path if translated_path is not None else backend.translated_path(original_path)
    translated_exe = executable_path(t_path)
    translated_cmd, run_env = effective_run(
        translated_exe, backend,
        profile=profile,
        mpi_backend=mpi_backend,
        hardware=hardware,
        mpi_launch=mpi_launch,
        compile_variant=compile_variant,
    )

    _log_execute_header(backend.name, translated_cmd, 1, env=run_env or None)

    try:
        trans_stdout, trans_stderr, _ = run_once(
            translated_exe, translated_cmd, env=run_env or None,
        )
    except ExecutionTimeoutError as exc:
        error_info = (
            f"The translated program exceeded the execution timeout "
            f"({EXECUTION_TIMEOUT_SECONDS} s).\n"
            f"This usually indicates an infinite loop, excessive computational load, or a "
            f"deadlock in the parallel code.\n\n"
            f"Error details:\n{exc}"
        )
        logger.info("  Result: execution timeout")
        return False, error_info
    except ExecutionError as exc:
        error_info = (
            f"The translated program crashed at runtime (non-zero exit code).\n"
            f"Common causes: segmentation fault, stack overflow, assertion failure, "
            f"MPI error, or unhandled exception.\n\n"
            f"Error details:\n{exc}"
        )
        logger.info("  Result: execution error (crash)")
        return False, error_info

    logger.info("  Result: execution OK")

    trans_checksum, _trans_text = parse_checksum(trans_stdout)

    if serial_checksum is not None and trans_checksum is None:
        error_info = "No 'Checksum:' line in stdout"
        logger.warning("  [WARNING] %s", error_info)
        return False, error_info

    if serial_checksum is not None and trans_checksum is not None:
        matches, delta = checksums_match(
            serial_checksum, trans_checksum, rtol, serial_checksum_text,
        )
        window = absolute_tolerance(serial_checksum, rtol, serial_checksum_text)
        logger.info("  Checksum serial:     %.6g", serial_checksum)
        logger.info("  Checksum %-10s %.6g", backend.name, trans_checksum)
        logger.info("  Absolute tolerance: %.3e", window)
        if matches:
            logger.info("  Absolute difference: %.3e  (within tolerance)", delta)
            return True, ""
        logger.warning(
            "  Absolute difference: %.3e  [WARNING] outside tolerance (%.3e)",
            delta, window,
        )
        error_info = (
            f"The translated program produced an incorrect numerical result.\n"
            f"The checksum differs from the serial reference beyond the allowed "
            f"absolute tolerance ({window:.3e}).\n\n"
            f"Serial checksum:     {serial_checksum:.6g}\n"
            f"Translated checksum: {trans_checksum:.6g}\n"
            f"Absolute difference: {delta:.3e}\n"
            f"Absolute tolerance:  {window:.3e}\n\n"
            f"This is typically caused by data races, incorrect reduction operations, "
            f"uninitialized memory, or incorrect parallelization boundaries."
        )
        if trans_stderr:
            error_info += f"\n\nProgram stderr:\n{trans_stderr}"
        return False, error_info

    # No checksum available — fall back to exact stdout comparison
    if serial_stdout == trans_stdout:
        logger.info("  Identical output (stdout)")
        return True, ""

    logger.warning("  [WARNING] stdout differs from the serial run")
    error_info = (
        f"The translated program produced different output than the serial reference.\n\n"
        f"Expected stdout (serial):\n{serial_stdout!r}\n\n"
        f"Actual stdout (translated):\n{trans_stdout!r}"
    )
    if trans_stderr:
        error_info += f"\n\nProgram stderr:\n{trans_stderr}"
    return False, error_info


def verify_and_benchmark(
    original_path: Path,
    backend: Backend,
    *,
    extra_flags: list[str] | None = None,
    translated_path: Path | None = None,
    profile: "ParadigmProfile | None" = None,
    mpi_backend: Backend | None = None,
    hardware: HardwareInfo | None = None,
    mpi_launch: MpiLaunch | None = None,
    compile_variant: str = "translated",
    rtol: float = CHECKSUM_RTOL,
    binding: OmpBinding | None = None,
) -> None:
    """Compile and run both the serial original and the translated version, then report results.

    Args:
        original_path:  Path to the original serial source file.
        backend:        The active backend.
        extra_flags:    Extra compiler flags.
        translated_path: Explicit path to the translated source/binary.  When
                         ``None``, the path is derived via ``backend.translated_path``.
        profile:        Accumulated paradigm profile for translated compilation/execution.
        mpi_backend:    MPI backend used when ``profile.mpi`` is True.
        compile_variant: Passed to execution helper to determine run mode.
        binding:        OpenMP placement for the translated run. When omitted,
                        the launch uses configuration A.
    """
    t_path = translated_path if translated_path is not None else backend.translated_path(original_path)
    serial_exe = executable_path(original_path)
    translated_exe = executable_path(t_path)

    logger.info("\n\n%s", "=" * 50)
    logger.info("  Verification: serial vs %s", backend.name)
    logger.info("%s", "=" * 50)

    logger.info("\n\n  Compiling the serial version…")
    ok, err = compile_file(
        original_path, backend, variant="serial", extra_flags=extra_flags,
    )
    if not ok:
        raise CompilationError("Could not compile the original file:", output=err)

    # Recompile translated with full profile flags
    logger.info("\n\n  Recompiling the translated version with the full profile…")
    ok, err = compile_file(
        t_path, backend, variant=compile_variant,
        extra_flags=extra_flags, profile=profile,
    )
    if not ok:
        raise CompilationError("Could not recompile the translated file:", output=err)

    runs = BENCHMARK_RUNS
    serial_cmd = backend.run_serial_cmd(serial_exe)
    translated_cmd, run_env = effective_run(
        translated_exe, backend,
        profile=profile,
        mpi_backend=mpi_backend,
        hardware=hardware,
        mpi_launch=mpi_launch,
        compile_variant=compile_variant,
        binding=binding,
    )

    serial_stdout, serial_stderr, serial_wall, serial_stdout_ms = benchmark(
        serial_exe, serial_cmd, runs, label="serial",
    )

    trans_stdout, trans_stderr, trans_wall, trans_stdout_ms = benchmark(
        translated_exe,
        translated_cmd,
        runs,
        label=backend.name,
        env=run_env or None,
    )

    logger.info("\n\n%s", "-" * 50)
    logger.info("  Correctness check")
    logger.info("%s", "-" * 50)

    serial_checksum, serial_checksum_text = parse_checksum(serial_stdout)
    trans_checksum, _trans_text = parse_checksum(trans_stdout)

    if serial_checksum is not None and trans_checksum is not None:
        matches, delta = checksums_match(
            serial_checksum, trans_checksum, rtol, serial_checksum_text,
        )
        window = absolute_tolerance(serial_checksum, rtol, serial_checksum_text)
        logger.info("  Checksum serial:     %.6g", serial_checksum)
        logger.info("  Checksum %-10s %.6g", backend.name, trans_checksum)
        logger.info("  Absolute tolerance: %.3e", window)
        if matches:
            logger.info("  Absolute difference: %.3e  (within tolerance)", delta)
        else:
            logger.warning(
                "  Absolute difference: %.3e  [WARNING] outside tolerance (%.3e)",
                delta, window,
            )
    else:
        logger.info("  (Checksum unavailable — exact stdout comparison)")
        stdout_match = serial_stdout == trans_stdout
        if stdout_match:
            logger.info("  Identical output (stdout)")
        else:
            logger.warning("  [WARNING] stdout differs:")
            logger.warning("     serial:    %r", serial_stdout)
            logger.warning("     %s: %r", backend.slug, trans_stdout)

    stderr_match = serial_stderr == trans_stderr
    if not stderr_match:
        logger.warning("  [WARNING] stderr differs:")
        logger.warning("     serial:    %r", serial_stderr)
        logger.warning("     %s: %r", backend.slug, trans_stderr)

    has_kernel = serial_stdout_ms is not None and trans_stdout_ms is not None
    if has_kernel:
        logger.info("\n\n%s", "-" * 50)
        logger.info("  Kernel")
        logger.info("%s", "-" * 50)
        logger.info("  Serial      (median of %s): %.3f ms", runs, serial_stdout_ms)
        logger.info("  %-10s (median of %s): %.3f ms", backend.name, runs, trans_stdout_ms)
        if trans_stdout_ms > 0:
            speedup_stdout = serial_stdout_ms / trans_stdout_ms
            logger.info("  Speedup: %.2fx", speedup_stdout)
            if speedup_stdout < 1.0:
                logger.info(
                    "  (The parallel kernel is slower. This can happen when the computational load is small)"
                )
        else:
            logger.info("  Speedup: N/A (translated time ~ 0)")

        logger.info("\n\n%s", "-" * 50)
        logger.info("  Process time (startup, init and communication)")
        logger.info("%s", "-" * 50)
        logger.info("  Serial      (median of %s): %.3f ms", runs, serial_wall * 1000)
        logger.info("  %-10s (median of %s): %.3f ms", backend.name, runs, trans_wall * 1000)
    else:
        logger.info("\n\n%s", "-" * 50)
        logger.info("  Process time (startup, init and communication)")
        logger.info("%s", "-" * 50)
        logger.info("  Serial      (median of %s): %.3f ms", runs, serial_wall * 1000)
        logger.info("  %-10s (median of %s): %.3f ms", backend.name, runs, trans_wall * 1000)
        if trans_wall > 0:
            speedup_wall = serial_wall / trans_wall
            logger.info("  Speedup: %.2fx", speedup_wall)
            if speedup_wall < 1.0:
                logger.info(
                    "  (The parallel version is slower. This can happen when the computational load is small)"
                )
        else:
            logger.info("  Speedup: N/A (translated time ~ 0)")
        logger.info("  (Kernel unavailable: the program does not print 'Tiempo de ejecucion ... ms')")

    _cleanup_executables(serial_exe, translated_exe)

    logger.info("\n%s\n", "=" * 50)
