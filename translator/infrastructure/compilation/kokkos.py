"""Compilación con Kokkos."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from translator.config.env import ensure_env_loaded, get_env
from translator.config.settings import ENV_PATH
from translator.exceptions import CompilerNotFoundError
from translator.infrastructure.compilation.common import executable_path, run_compile


def _kokkos_flags() -> tuple[str, list[str], list[str]]:
    ensure_env_loaded()

    env_cxx = get_env("KOKKOS_CXX")
    env_cxxflags = get_env("KOKKOS_CXXFLAGS")
    env_ldflags = get_env("KOKKOS_LDFLAGS")

    if env_cxxflags or env_ldflags:
        cxx = env_cxx or "g++"
        return cxx, env_cxxflags.split(), env_ldflags.split()

    if shutil.which("kokkos_config"):
        try:
            cxx = subprocess.check_output(
                ["kokkos_config", "--compiler"], text=True
            ).strip() or (env_cxx or "g++")
            cxxflags = subprocess.check_output(
                ["kokkos_config", "--cxxflags"], text=True
            ).split()
            ldflags = subprocess.check_output(
                ["kokkos_config", "--ldflags"], text=True
            ).split()
            return cxx, cxxflags, ldflags
        except subprocess.CalledProcessError:
            pass

    raise CompilerNotFoundError(
        f"Could not detect the Kokkos installation.\n"
        f"        Expected .env at: {ENV_PATH}\n"
        f"        Set KOKKOS_CXXFLAGS and KOKKOS_LDFLAGS in .env,\n"
        f"        or make sure kokkos_config is on PATH."
    )


def compile_kokkos(
    source_path: Path,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    cxx, cxxflags, ldflags = _kokkos_flags()
    extra = extra_flags or []
    exe = executable_path(source_path)
    cmd = [cxx, *cxxflags, *extra, str(source_path), "-o", str(exe), *ldflags]
    return run_compile(cmd, label="Kokkos")
