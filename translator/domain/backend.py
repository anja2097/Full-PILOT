"""Backend domain entity — a pure data class with no infrastructure imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Backend:
    name: str
    slug: str
    prompts_dir: Path
    extension: str
    run_prefix: list[str] = field(default_factory=list)
    run_env_vars: dict[str, str] = field(default_factory=dict)

    def translated_path(self, source_path: Path) -> Path:
        """Return the path for the translated file.

        When ``extension`` is empty the source file extension is preserved.
        """
        ext = self.extension or source_path.suffix
        return source_path.with_name(f"{source_path.stem}_{self.slug}{ext}")

    def run_cmd(self, exe_path: Path) -> list[str]:
        """Return the command to run the translated binary."""
        return self.run_prefix + [str(exe_path)]

    def run_serial_cmd(self, exe_path: Path) -> list[str]:
        """Return the command to run the serial binary (never prefixed)."""
        return [str(exe_path)]

    def run_env(self) -> dict[str, str]:
        """Return extra environment variables needed to run the translated binary."""
        return self.run_env_vars
