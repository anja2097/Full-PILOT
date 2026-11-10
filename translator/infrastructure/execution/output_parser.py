"""Parsing of program stdout: checksum and self-reported execution time."""

from __future__ import annotations

import re

_STDOUT_TIME_RE = re.compile(r"Tiempo de ejecucion[^:]*:\s*([\d.]+)\s*ms", re.IGNORECASE)
_CHECKSUM_RE = re.compile(r"Checksum:\s*([\d.eE+\-]+)", re.IGNORECASE)


def parse_checksum(stdout: str) -> tuple[float | None, str | None]:
    """Extract ``(value, printed token)`` from a 'Checksum: <value>' line.

    The token keeps the digits the program printed, so the comparison can
    allow one unit in the last printed place. Returns ``(None, None)`` if
    the line is missing or the token is not a float.
    """
    for line in stdout.splitlines():
        m = _CHECKSUM_RE.search(line)
        if m:
            token = m.group(1)
            try:
                return float(token), token
            except ValueError:
                return None, None
    return None, None


def printed_ulp(token: str | None) -> float:
    """Unit of the last printed digit of a checksum token.

    ``268527041.138949`` and ``0.001234`` are ``1e-6``. ``2.68527e+08`` is
    ``1e+03``. An empty token contributes nothing (``0``).
    """
    if not token:
        return 0.0
    text = token.strip()
    scientific = re.fullmatch(r"[+-]?\d+(?:\.(\d+))?[eE]([+-]?\d+)", text)
    if scientific:
        fraction = scientific.group(1) or ""
        exponent = int(scientific.group(2))
        return 10.0 ** (exponent - len(fraction))
    fixed = re.fullmatch(r"[+-]?\d+\.(\d+)", text)
    if fixed:
        return 10.0 ** (-len(fixed.group(1)))
    if re.fullmatch(r"[+-]?\d+", text):
        return 1.0
    return 0.0


def absolute_tolerance(
    reference: float,
    rtol: float,
    reference_text: str | None = None,
) -> float:
    """Absolute acceptance window, including one printed unit and a rounding slack."""
    ulp = printed_ulp(reference_text)
    if reference == 0.0:
        base = max(rtol, ulp)
    else:
        base = max(rtol * abs(reference), ulp)
    return base * (1.0 + 1e-9)


def checksums_match(
    reference: float,
    actual: float,
    rtol: float,
    reference_text: str | None = None,
) -> tuple[bool, float]:
    """Compare checksums.

    Returns ``(matches, absolute_delta)``. The match window is
    ``max(rtol * |reference|, printed ulp)``, or ``max(rtol, printed ulp)``
    when the reference is 0, times ``1 + 1e-9``.
    """
    delta = abs(reference - actual)
    return delta <= absolute_tolerance(reference, rtol, reference_text), delta


def first_stdout_diff(expected: str, actual: str) -> str:
    """Describe the first pair of stdout lines that differ."""
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    count = max(len(expected_lines), len(actual_lines))
    for index in range(count):
        left = expected_lines[index] if index < len(expected_lines) else ""
        right = actual_lines[index] if index < len(actual_lines) else ""
        if left != right:
            return f"Stdout distinto: expected {left!r}, got {right!r}"
    return "Stdout distinto: expected '' , got ''"


def parse_stdout_time(stdout: str) -> float | None:
    """Extract the time in ms from a 'Tiempo de ejecucion ...: X.XX ms' line.

    Returns None if the line is not present or the value cannot be parsed.
    """
    for line in stdout.splitlines():
        m = _STDOUT_TIME_RE.search(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None
