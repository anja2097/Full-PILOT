"""Syntax-only compiles of SIMD branches that ``-march=native`` does not build."""

from __future__ import annotations

import subprocess
from pathlib import Path

_MARCH_FLAGS = {
    "x86-64-v4": ["-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl", "-mfma"],
    "x86-64-v3": ["-mavx2", "-mfma"],
    "x86-64-v2": ["-mno-avx"],
}

_BRANCHES = (
    ("AVX-512", "x86-64-v4"),
    ("AVX2", "x86-64-v3"),
    ("scalar", "x86-64-v2"),
)

_ISA_TOKENS = frozenset({
    "-mavx", "-mavx2", "-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl",
    "-mfma", "-mno-avx", "-mno-avx2", "-mno-avx512f",
})

_probe_cache: dict[tuple[str, str], bool] = {}


def source_has_simd_branches(source: str) -> bool:
    return "__AVX512F__" in source or "__AVX2__" in source


def _without_output_and_isa(argv: list[str]) -> list[str]:
    kept: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "-o":
            skip_next = True
            continue
        if token in _ISA_TOKENS or token.startswith("-march=") or token.startswith("-mtune="):
            continue
        if token.startswith("-fopt-info"):
            continue
        kept.append(token)
    return kept


def _accepts_march(compiler: str, march: str) -> bool:
    key = (compiler, march)
    cached = _probe_cache.get(key)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [compiler, f"-march={march}", "-fsyntax-only", "-x", "c", "-"],
            input="int main(void){return 0;}\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        _probe_cache[key] = False
        return False
    ok = result.returncode == 0
    _probe_cache[key] = ok
    return ok


def check_simd_branches(source_path: Path, compile_argv: list[str]) -> tuple[bool, str]:
    """Compile each preprocessor branch for syntax only.

    *compile_argv* is the stage command (it may include ``-march=native`` and
    ``-o``). A failure is reported as a compilation error naming the branch.
    """
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, str(exc)
    if not source_has_simd_branches(source):
        return True, ""
    if not compile_argv:
        return False, "No compiler command for SIMD syntax check"

    compiler = compile_argv[0]
    base = _without_output_and_isa(compile_argv)
    # Drop the original source path; it is re-added once, after the ISA flags.
    base = [token for token in base if token != str(source_path)]
    if "-fsyntax-only" not in base:
        base.insert(1, "-fsyntax-only")

    errors: list[str] = []
    for branch_name, march in _BRANCHES:
        if _accepts_march(compiler, march):
            isa = [f"-march={march}"]
            shown = f"-march={march}"
        else:
            isa = list(_MARCH_FLAGS[march])
            shown = " ".join(isa)
        cmd = [*base, *isa, str(source_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{branch_name} branch (compiled with {shown}) fails: {exc}")
            continue
        if result.returncode != 0:
            output = (result.stdout + result.stderr).strip()
            errors.append(
                f"{branch_name} branch (compiled with {shown}) fails:\n{output}"
            )
    if errors:
        return False, "\n".join(errors)
    return True, ""
