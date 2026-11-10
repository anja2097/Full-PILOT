"""Low-level process execution with timeout and error handling."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from translator.config.settings import EXECUTION_TIMEOUT_SECONDS
from translator.exceptions import ExecutionError, ExecutionTimeoutError

logger = logging.getLogger(__name__)


def run_once(
    exe_path: Path,
    cmd: list[str],
    env: dict[str, str] | None = None,
) -> tuple[str, str, float]:
    """Run a command once and return (stdout, stderr, elapsed_seconds).

    Raises:
        ExecutionTimeoutError: If the process exceeds EXECUTION_TIMEOUT_SECONDS.
        ExecutionError: If the process exits with a non-zero return code.
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=exe_path.parent,
            env=run_env,
            timeout=EXECUTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.info("  Result: execution timed out")
        raise ExecutionTimeoutError(
            f"Execution exceeded {EXECUTION_TIMEOUT_SECONDS} s: {exe_path}\n"
            f"        Possible infinite loop or a program that is too slow."
        )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        logger.info("  Result: execution error")
        raise ExecutionError(
            f"The executable failed: {exe_path}\n{result.stderr or result.stdout}",
            returncode=result.returncode,
            stderr=result.stderr or "",
        )

    return result.stdout, result.stderr, elapsed
