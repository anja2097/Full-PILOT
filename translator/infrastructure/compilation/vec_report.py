"""GCC vectorization report for the input of an AVX-aware stage."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

_TIMER_RE = re.compile(r"clock_gettime|std::chrono|omp_get_wtime|MPI_Wtime")
_OPT_FLAGS = ("-fopt-info-vec-optimized", "-fopt-info-vec-missed")


def timed_region(source: str) -> tuple[int, int] | None:
    """Line span covering every recognizable timer call, if there are at least two."""
    lines = [
        index
        for index, line in enumerate(source.splitlines(), start=1)
        if _TIMER_RE.search(line)
    ]
    if len(lines) < 2:
        return None
    return lines[0], lines[-1]


def select_vec_lines(
    stderr: str,
    source_name: str,
    source_text: str,
    *,
    limit: int = 40,
) -> list[str]:
    """Lines that mention *source_name* and a loop, timed-region lines first."""
    span = timed_region(source_text)
    seen: set[str] = set()
    inside: list[str] = []
    outside: list[str] = []
    line_no = re.compile(re.escape(source_name) + r":(\d+)")
    for raw in stderr.splitlines():
        line = raw.strip()
        if not line or source_name not in line or "loop" not in line.lower():
            continue
        if line in seen:
            continue
        seen.add(line)
        match = line_no.search(line)
        number = int(match.group(1)) if match else None
        if span is not None and number is not None and span[0] <= number <= span[1]:
            inside.append(line)
        else:
            outside.append(line)
    return (inside + outside)[:limit]


def _with_report_flags(argv: list[str], output_path: Path) -> list[str]:
    cmd: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "-o":
            skip_next = True
            continue
        cmd.append(token)
    for flag in _OPT_FLAGS:
        if flag not in cmd:
            cmd.append(flag)
    cmd.extend(["-o", str(output_path)])
    return cmd


def collect_vec_report(source_path: Path, compile_argv: list[str]) -> str | None:
    """Compile *source_path* only to capture ``-fopt-info-vec`` diagnostics.

    Returns the filtered report, or None when the compile fails or nothing
    mentions a loop. The temporary binary is removed.
    """
    if not compile_argv:
        return None
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except OSError:
        return None

    fd, name = tempfile.mkstemp(prefix="pilot-vec-")
    os.close(fd)
    output = Path(name)
    try:
        cmd = _with_report_flags(compile_argv, output)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        lines = select_vec_lines(
            result.stderr + "\n" + result.stdout,
            source_path.name,
            source_text,
        )
        if not lines:
            return None
        return "\n".join(lines)
    finally:
        if output.is_file():
            output.unlink()


def vec_report_block(source_path: Path, compile_argv: list[str]) -> str | None:
    body = collect_vec_report(source_path, compile_argv)
    if not body:
        return None
    return "## GCC vectorization report for the input (-fopt-info-vec)\n" + body
