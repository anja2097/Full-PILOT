"""Harness context prepended to translation and optimization prompts."""

from __future__ import annotations

from translator.config.settings import BENCHMARK_RUNS, CHECKSUM_RTOL, format_tolerance
from translator.infrastructure.hardware import (
    HardwareInfo,
    MpiLaunch,
    mpirun_command,
    omp_bindings,
)

_OMP_NOTE = (
    "The harness measures both and keeps the faster. With configuration A two "
    "threads share each core's L1d, L2 and FMA units; with B each thread owns a "
    "core. Size per-thread blocks so they work for B; they must stay correct for A."
)

DUAL_PROMPT_STAGES = frozenset({"openmp", "optopenmp", "mpi", "optmpi"})


def verification_block(rtol: float = CHECKSUM_RTOL) -> str:
    shown = format_tolerance(rtol)
    return (
        "## Verification\n"
        "The harness compares the program's `Checksum:` line with the serial reference "
        f"using a relative tolerance of {shown} (absolute if the reference is 0). "
        "If the program prints no checksum, stdout must match exactly. "
        "Reassociating sums (SIMD lanes, multiple accumulators, OpenMP/MPI reductions) "
        "is acceptable within this tolerance; do NOT force left-to-right summation order."
    )


def openmp_launch_commands(
    hardware: HardwareInfo | None,
    mpi_launch: MpiLaunch | None,
    *,
    per_rank: bool,
) -> list[tuple[str, str]]:
    """Exact launch lines the harness will measure, with ``<program>`` as the binary."""
    bindings = omp_bindings(hardware, per_rank=per_rank)
    lines: list[tuple[str, str]] = []
    for binding in bindings:
        if per_rank:
            if mpi_launch is None:
                continue
            cmd = " ".join(mpirun_command("<program>", mpi_launch, binding))
        else:
            cmd = (
                f"OMP_NUM_THREADS={binding.threads} OMP_PLACES={binding.places} "
                f"OMP_PROC_BIND={binding.proc_bind} <program>"
            )
        lines.append((binding.name, cmd))
    return lines


def openmp_launch_block(commands: list[tuple[str, str]]) -> str:
    rows = "\n".join(f"- {name}: `{cmd}`" for name, cmd in commands)
    return f"## OpenMP launch\n\n{rows}\n\n{_OMP_NOTE}"


def performance_block(
    *,
    process_ms: float | None,
    kernel_ms: float | None,
    launch_line: str,
    serial_kernel_ms: float | None,
    runs: int = BENCHMARK_RUNS,
) -> str | None:
    """Input timing for an optimization prompt. None when there is no process time."""
    if process_ms is None:
        return None
    lines = ["## Performance of the input"]
    if kernel_ms is not None:
        lines.append(
            f"- Kernel time of the input program: {kernel_ms:.1f} ms "
            f"(median of {runs} runs after a warm-up), measured with: {launch_line}"
        )
    lines.append(f"- Process wall time: {process_ms:.1f} ms")
    if serial_kernel_ms is not None and kernel_ms is not None and kernel_ms > 0:
        speedup = serial_kernel_ms / kernel_ms
        lines.append(
            f"- Serial baseline kernel time: {serial_kernel_ms:.1f} ms "
            f"(current speedup vs serial: {speedup:.2f}x)"
        )
    return "\n".join(lines)


def compose_stage_prompt(
    body: str,
    *,
    hardware: HardwareInfo | None,
    mpi_launch: MpiLaunch | None,
    include_mpi_launch: bool,
    logical_name: str,
    profile_mpi: bool,
    rtol: float = CHECKSUM_RTOL,
    performance: str | None = None,
    vec_report: str | None = None,
) -> str:
    """Assemble machine, launch, verification, performance and the stage prompt."""
    parts: list[str] = []
    if hardware is not None:
        parts.append(hardware.to_prompt_block().strip())
    if include_mpi_launch and mpi_launch is not None and mpi_launch.supported:
        parts.append(mpi_launch.to_prompt_block().strip())
    if logical_name.lower() in DUAL_PROMPT_STAGES:
        commands = openmp_launch_commands(
            hardware, mpi_launch, per_rank=profile_mpi,
        )
        if commands:
            parts.append(openmp_launch_block(commands))
    parts.append(verification_block(rtol))
    if performance:
        parts.append(performance.strip())
    if vec_report:
        parts.append(vec_report.strip())
    parts.append(body)
    return "\n\n".join(parts)
