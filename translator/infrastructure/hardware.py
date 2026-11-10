"""Hardware information capture for optimization prompts."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ISA flags of interest for HPC optimisation decisions.
_ISA_KEYWORDS: frozenset[str] = frozenset({
    "avx", "avx2",
    "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl", "avx512vnni",
    "fma",
    "sse4_1", "sse4_2", "ssse3",
})


@dataclass(frozen=True)
class HardwareInfo:
    """Parsed hardware characteristics relevant to optimisation decisions.

    All string fields default to ``"N/A"`` when the lscpu output does not
    contain the corresponding line.  Integer fields default to ``1``.

    Note on cache labels: lscpu may report aggregated cache sizes (e.g.
    ``"4 MiB (128 instances)"``).  The raw string is preserved here so that
    prompts can present the value faithfully rather than incorrectly claiming
    it is "per core".
    """

    l1d_cache: str          # e.g. "32 KiB" or "4 MiB (128 instances)"
    l1i_cache: str
    l2_cache: str
    l3_cache: str
    cache_line_bytes: int   # typically 64; hardcoded since lscpu rarely exposes it
    cores_per_socket: int
    threads_per_core: int
    sockets: int
    numa_nodes: int
    isa_flags: tuple[str, ...]  # immutable; subset of _ISA_KEYWORDS present on host
    l3_detail: str | None = None  # "16 MB per instance, shared by 4 cores; ..."

    @property
    def total_cores(self) -> int:
        return self.sockets * self.cores_per_socket

    @property
    def total_threads(self) -> int:
        return self.total_cores * self.threads_per_core

    def to_prompt_block(self) -> str:
        """Format as a Markdown block to prepend to Opt* prompts and to the first turn of a translation stage.

        Cache sizes are reproduced verbatim from lscpu (not labelled "per core")
        to avoid misleading the LLM when lscpu reports aggregate totals.
        """
        flags_str = " ".join(self.isa_flags) if self.isa_flags else "(none detected)"
        if self.l3_detail:
            l3_line = f"- L3: {self.l3_detail}\n"
        else:
            l3_line = f"- L3 cache:  {self.l3_cache}\n"
        return (
            "## Target machine\n"
            "\n"
            f"- L1d cache: {self.l1d_cache}\n"
            f"- L1i cache: {self.l1i_cache}\n"
            f"- L2 cache:  {self.l2_cache}\n"
            f"{l3_line}"
            f"- Cache line: {self.cache_line_bytes} bytes\n"
            f"- Cores: {self.total_cores} "
            f"({self.sockets} socket × {self.cores_per_socket} cores × "
            f"{self.threads_per_core} threads/core)\n"
            f"- NUMA nodes: {self.numa_nodes}\n"
            f"- ISA flags: {flags_str}\n"
        )


MPI_HARDWARE_MESSAGE = (
    "This hardware is not a good fit for MPI. "
    "Try a different pipeline."
)


@dataclass(frozen=True)
class MpiLaunch:
    """Colocación híbrida MPI+OpenMP para Open MPI 4.1.1.

    ``supported`` es False solo con un nodo y un socket. En ese caso
    no hay comando: el pipeline debe avisar y no ejecutar nada.
    """

    supported: bool
    nodes: int
    numa_per_node: int
    ranks: int
    omp_threads: int
    map_by: str
    placement: str

    def to_prompt_block(self) -> str:
        """Describe the rank count and placement for an MPI or OptMPI prompt."""
        return (
            "## MPI launch\n"
            "\n"
            f"- Nodes: {self.nodes}\n"
            f"- NUMA domains per node: {self.numa_per_node}\n"
            f"- Total ranks: {self.ranks}\n"
            f"- OpenMP threads per rank: {self.omp_threads}\n"
            f"- Placement: {self.placement}\n"
            "- Do not hardcode the rank count. Read it with MPI_Comm_size.\n"
        )


def slurm_node_count(environ: dict[str, str] | None = None) -> int:
    """Nodes in the current Slurm allocation.

    ``SLURM_JOB_NUM_NODES`` wins over ``SLURM_NNODES``. Outside Slurm, or
    when both values are missing or invalid, the allocation is one node.
    """
    import os

    env = os.environ if environ is None else environ
    for key in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        raw = env.get(key, "").strip()
        if not raw:
            continue
        try:
            count = int(raw)
        except ValueError:
            continue
        if count >= 1:
            return count
    return 1


def _local_topology(hw: HardwareInfo | None) -> tuple[int, int, int, int]:
    """Return (NUMA domains, sockets, cores per socket, threads per core)."""
    import os

    if hw is None:
        logical = max(os.cpu_count() or 1, 1)
        return 1, 1, logical, 1
    return (
        max(hw.numa_nodes, 1),
        max(hw.sockets, 1),
        max(hw.cores_per_socket, 1),
        max(hw.threads_per_core, 1),
    )


def decide_mpi_launch(
    hw: HardwareInfo | None,
    *,
    nodes: int | None = None,
    environ: dict[str, str] | None = None,
) -> MpiLaunch:
    """Choose ranks and OpenMP threads for a hybrid MPI+OpenMP launch.

    One rank per socket: ``map-by ppr:1:socket:PE=<cores per socket>``.
    OpenMP threads per rank are ``cores per socket × threads per core``.
    One node with a single socket is not a valid MPI target.
    """
    if nodes is None:
        nodes = slurm_node_count(environ)
    nodes = max(nodes, 1)
    numa_per_node, sockets, cores_per_socket, threads_per_core = _local_topology(hw)

    if nodes == 1 and sockets == 1:
        return MpiLaunch(
            supported=False,
            nodes=nodes,
            numa_per_node=numa_per_node,
            ranks=0,
            omp_threads=0,
            map_by="",
            placement="",
        )

    return MpiLaunch(
        supported=True,
        nodes=nodes,
        numa_per_node=numa_per_node,
        ranks=nodes * sockets,
        omp_threads=cores_per_socket * threads_per_core,
        map_by=f"ppr:1:socket:PE={cores_per_socket}",
        placement="1 rank por socket",
    )


def include_mpi_launch_prompt(backend_slug: str, compile_variant: str) -> bool:
    """True for MPI and OptMPI prompts, not for a serial stage that borrows MPI."""
    return backend_slug == "mpi" and compile_variant != "serial"


def prepend_hardware_prompt(
    prompt: str,
    hardware: HardwareInfo | None,
    launch: MpiLaunch | None = None,
    *,
    include_launch: bool = False,
) -> str:
    """Prepend the machine block and, when requested, the MPI launch block."""
    parts: list[str] = []
    if hardware is not None:
        parts.append(hardware.to_prompt_block())
    if include_launch and launch is not None and launch.supported:
        parts.append(launch.to_prompt_block())
    if not parts:
        return prompt
    return "\n\n".join(parts) + "\n\n" + prompt


@dataclass(frozen=True)
class OmpBinding:
    """One OpenMP placement the harness may measure."""

    name: str          # "A" or "B"
    threads: int
    places: str
    proc_bind: str


def omp_bindings(hw: HardwareInfo | None, *, per_rank: bool) -> list[OmpBinding]:
    """Configuration A, plus B when SMT gives two threads per core.

    A puts a thread on every logical CPU (``places=threads``, ``bind=close``).
    B puts one thread on every physical core (``places=cores``, ``bind=spread``).
    With ``per_rank`` the counts are for one MPI rank (one socket). Otherwise
    they cover the whole node. When there is no SMT, A and B match and only A
    is returned.
    """
    _numa, sockets, cores, threads_per = _local_topology(hw)
    if per_rank:
        a_threads = cores * threads_per
        b_threads = cores
    else:
        a_threads = sockets * cores * threads_per
        b_threads = sockets * cores
    configs = [OmpBinding("A", a_threads, "threads", "close")]
    if threads_per > 1:
        configs.append(OmpBinding("B", b_threads, "cores", "spread"))
    return configs


def mpirun_command(
    exe: str,
    launch: MpiLaunch,
    binding: OmpBinding | None = None,
) -> list[str]:
    """Open MPI 4.1.1 command for *launch*.

    *binding* selects the OpenMP thread placement. The default is configuration
    A: ``OMP_NUM_THREADS`` from the launch and ``places=threads``.

    Raises:
        MpiHardwareError: if *launch* is the unsupported single-socket node.
    """
    from translator.exceptions import MpiHardwareError

    if not launch.supported:
        raise MpiHardwareError(MPI_HARDWARE_MESSAGE)
    if binding is None:
        threads = launch.omp_threads
        places = "threads"
        proc_bind = "close"
    else:
        threads = binding.threads
        places = binding.places
        proc_bind = binding.proc_bind
    return [
        "mpirun",
        "-np", str(launch.ranks),
        "--oversubscribe",
        "--map-by", launch.map_by,
        "--bind-to", "core",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^openib",
        "-x", f"OMP_NUM_THREADS={threads}",
        "-x", f"OMP_PLACES={places}",
        "-x", f"OMP_PROC_BIND={proc_bind}",
        exe,
    ]


def openmp_thread_count(hw: HardwareInfo | None) -> int:
    """Hilos lógicos para un OpenMP puro, sin MPI.

    Con ``lscpu`` es ``sockets × cores × threads/core``. Sin él se usa
    ``os.cpu_count()``. Si tampoco hay recuento, un hilo.
    """
    import os

    if hw is not None:
        return max(hw.total_threads, 1)
    return max(os.cpu_count() or 1, 1)


def normalize_cache_size(raw: str) -> str:
    """Turn ``16384K`` or ``16 MiB`` into ``16 MB`` when the value is a whole number of MB."""
    text = re.sub(r"\s*\(.*\)\s*$", "", raw.strip())
    match = re.match(r"^([\d.]+)\s*([kKmMgG])(?:i?[bB])?$", text)
    if not match:
        return raw.strip()
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "k":
        mb = value / 1024.0
    elif unit == "m":
        mb = value
    else:
        mb = value * 1024.0
    if abs(mb - round(mb)) < 1e-6:
        return f"{int(round(mb))} MB"
    return f"{mb:g} MB"


def describe_l3_instance(
    l3_raw: str,
    lscpu_p: str | None,
    sysfs_size: str | None = None,
) -> str | None:
    """Per-instance L3 line, or None when the parseable topology is missing.

    The returned text has the form
    ``16 MB per instance, shared by 4 cores; 4 instances per socket``.
    """
    if not lscpu_p or not lscpu_p.strip():
        return None

    cores_by_l3: dict[str, set[tuple[int, int]]] = {}
    l3_by_socket: dict[int, set[str]] = {}
    saw_l3 = False
    for line in lscpu_p.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(",")
        if len(parts) < 5:
            continue
        try:
            core = int(parts[1])
            socket = int(parts[2])
        except ValueError:
            continue
        l3_id = parts[4].strip().split(":")[-1]
        if not l3_id:
            continue
        saw_l3 = True
        cores_by_l3.setdefault(l3_id, set()).add((socket, core))
        l3_by_socket.setdefault(socket, set()).add(l3_id)

    if not saw_l3 or not cores_by_l3 or not l3_by_socket:
        return None

    shared = {len(cores) for cores in cores_by_l3.values()}
    per_socket = {len(ids) for ids in l3_by_socket.values()}
    if len(shared) != 1 or len(per_socket) != 1:
        return None

    size = _l3_instance_size(l3_raw, sysfs_size)
    cores = next(iter(shared))
    instances = next(iter(per_socket))
    return (
        f"{size} per instance, shared by {cores} cores; "
        f"{instances} instances per socket"
    )


def _l3_instance_size(l3_raw: str, sysfs_size: str | None) -> str:
    if sysfs_size and sysfs_size.strip():
        return normalize_cache_size(sysfs_size)
    match = re.search(
        r"([\d.]+)\s*([kKmMgG](?:i?[bB])?)\s*\(\s*(\d+)\s+instances?\s*\)",
        l3_raw,
        re.IGNORECASE,
    )
    if match:
        total = float(match.group(1))
        count = int(match.group(3))
        if count > 0:
            each = total / count
            return normalize_cache_size(f"{each:g} {match.group(2)}")
    return normalize_cache_size(l3_raw)


def stage_resources(
    profile: "object | None",
    hw: HardwareInfo | None,
    launch: MpiLaunch | None,
    compile_variant: str,
) -> tuple[int, int]:
    """Return ``(nodes, physical_cores)`` used by a stage.

    Serial and AVX-only stages use one core on one node. OpenMP without MPI
    uses every physical core of the node. MPI uses every physical core of
    every Slurm node.
    """
    mpi = bool(getattr(profile, "mpi", False))
    openmp = bool(getattr(profile, "openmp", False))
    if compile_variant == "serial" or not (mpi or openmp):
        return 1, 1
    _numa, sockets, cores, _threads = _local_topology(hw)
    if mpi:
        nodes = launch.nodes if launch is not None else slurm_node_count()
        return nodes, nodes * sockets * cores
    return 1, sockets * cores


def _sysfs_l3_size() -> str | None:
    from pathlib import Path

    path = Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None


def _capture_lscpu_p() -> str | None:
    try:
        result = subprocess.run(
            ["lscpu", "-p=CPU,CORE,SOCKET,NODE,CACHE"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_lscpu_env(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def capture_hardware_info() -> HardwareInfo | None:
    """Run ``lscpu`` with ``LC_ALL=C`` and parse the relevant fields.

    Returns ``None`` if ``lscpu`` is not available, times out, or parsing
    fails for all fields.  The rest of the pipeline continues unaffected.
    """
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_lscpu_env(),
        )
        if result.returncode != 0:
            logger.warning("lscpu exited with code %s", result.returncode)
            return None
    except FileNotFoundError:
        logger.warning("lscpu not found in PATH; hardware information will be skipped.")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("lscpu timed out; hardware information will be skipped.")
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("Unexpected error while running lscpu: %s", exc)
        return None

    info = _parse_lscpu(result.stdout)
    if info is None:
        return None
    detail = describe_l3_instance(info.l3_cache, _capture_lscpu_p(), _sysfs_l3_size())
    if detail is None:
        return info
    return HardwareInfo(
        l1d_cache=info.l1d_cache,
        l1i_cache=info.l1i_cache,
        l2_cache=info.l2_cache,
        l3_cache=info.l3_cache,
        cache_line_bytes=info.cache_line_bytes,
        cores_per_socket=info.cores_per_socket,
        threads_per_core=info.threads_per_core,
        sockets=info.sockets,
        numa_nodes=info.numa_nodes,
        isa_flags=info.isa_flags,
        l3_detail=detail,
    )


def _lscpu_env() -> dict[str, str]:
    """Return an environment dict with LC_ALL=C to get English lscpu output."""
    import os
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    return env


def _parse_lscpu(output: str) -> HardwareInfo | None:
    """Parse lscpu output into a :class:`HardwareInfo` instance.

    Returns ``None`` only if the output is completely empty (not just missing
    some fields — missing fields get sensible defaults).
    """
    if not output.strip():
        return None

    def _get(pattern: str, default: str = "N/A") -> str:
        m = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else default

    def _get_int(pattern: str, default: int = 1) -> int:
        raw = _get(pattern, str(default))
        # Strip trailing qualifiers like "(128 instances)" if present
        first_token = raw.split()[0] if raw.split() else str(default)
        try:
            return int(first_token)
        except ValueError:
            return default

    # Extract ISA flags from the "Flags:" line
    flags_line = ""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("flags:"):
            flags_line = stripped.split(":", 1)[1] if ":" in stripped else ""
            break
    all_flags = flags_line.lower().split()
    isa_flags = tuple(sorted(f for f in all_flags if f in _ISA_KEYWORDS))

    return HardwareInfo(
        l1d_cache=_get(r"^L1d\s+cache\s*:\s*(.+)$"),
        l1i_cache=_get(r"^L1i\s+cache\s*:\s*(.+)$"),
        l2_cache=_get(r"^L2\s+cache\s*:\s*(.+)$"),
        l3_cache=_get(r"^L3\s+cache\s*:\s*(.+)$"),
        cache_line_bytes=64,  # Almost universal; lscpu rarely exposes this
        cores_per_socket=_get_int(r"^Core\(s\) per socket\s*:\s*(\d+)$"),
        threads_per_core=_get_int(r"^Thread\(s\) per core\s*:\s*(\d+)$"),
        sockets=_get_int(r"^Socket\(s\)\s*:\s*(\d+)$"),
        numa_nodes=_get_int(r"^NUMA node\(s\)\s*:\s*(\d+)$"),
        isa_flags=isa_flags,
    )
