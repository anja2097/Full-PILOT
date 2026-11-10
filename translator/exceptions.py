"""Domain exceptions for the translator pipeline.

All public exceptions inherit from TranslatorError so callers can catch
the whole family with a single ``except TranslatorError`` clause.
"""

from __future__ import annotations


class TranslatorError(Exception):
    """Base class for all translator errors."""


class CompilerNotFoundError(TranslatorError):
    """Raised when the required compiler binary is not found in PATH."""


class CompilationError(TranslatorError):
    """Raised when compilation fails.

    Attributes:
        output: Raw compiler stdout/stderr output.
    """

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


class LLMError(TranslatorError):
    """Raised when the LLM API returns an unrecoverable error.

    Attributes:
        code: HTTP status code or API error code.
    """

    def __init__(self, message: str, code: int | str = 0) -> None:
        super().__init__(message)
        self.code = code


class LLMEmptyResponseError(LLMError):
    """Raised when the LLM returns an empty response or no extractable code."""


class ExecutionError(TranslatorError):
    """Raised when a compiled binary exits with a non-zero return code.

    Attributes:
        returncode: Process exit status. Negative values are signals.
        stderr: Captured standard error (may be empty).
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class ExecutionTimeoutError(ExecutionError):
    """Raised when a binary exceeds the execution timeout."""


class MaxRetriesExhaustedError(TranslatorError):
    """Raised when the compilation-fix loop exhausts all retry attempts."""


class MpiHardwareError(TranslatorError):
    """Raised when the allocation cannot run an MPI pipeline."""


class PipelineStageError(TranslatorError):
    """Raised when a pipeline stage fails after exhausting its retries.

    Attributes:
        stage_name: Name of the backend stage that failed.
        stage_index: 0-based position of the failing stage in the pipeline.
    """

    def __init__(self, message: str, stage_name: str, stage_index: int) -> None:
        super().__init__(message)
        self.stage_name = stage_name
        self.stage_index = stage_index
