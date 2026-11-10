"""Constantes y resolución de configuración del traductor."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

MODELS: dict[str, str] = {
    "qwen3-coder": "qwen/qwen3-coder",
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "kimi-k2.6": "moonshotai/kimi-k2.6:free",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemma-4-31b": "google/gemma-4-31b-it",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "nemotron-3-nano": "nvidia/nemotron-3-nano-30b-a3b:free",
    "sonnet-5": "anthropic/claude-sonnet-5",
    "kimi-k3": "moonshotai/kimi-k3",
    "nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "deepseek-v4-flash": "deepseek/deepseek-v4.1-flash",
    "muse-spark-1.3": "meta/muse-spark-1.3"
}

HF_MODELS: dict[str, str] = {
    "qwen3-coder": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "kimi-k3": "moonshotai/Kimi-K3",
}

DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_BACKEND_NAME = "OpenMP"

THINKING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")

MAX_RETRIES = 5
BENCHMARK_RUNS = 11
EXECUTION_TIMEOUT_SECONDS = 300
CHECKSUM_RTOL = 1e-6  # tolerancia relativa; la comparación también admite una unidad impresa


def format_tolerance(rtol: float) -> str:
    """Compact scientific notation, e.g. ``1e-6`` rather than ``1e-06``."""
    return f"{rtol:.0e}".replace("e-0", "e-").replace("e+0", "e+")

# ── Mix of LLMs (etapas de optimización) ──────────────────────────────────────
# Agent models are tried in order; each produces one optimised variant.
MOL_DEFAULT_AGENT_MODELS: list[str] = [
    "qwen/qwen3-coder",
    "deepseek-v4-flash",
    "openai/gpt-oss-120b",
]
MOL_DEFAULT_AUDIT_MODEL: str = "anthropic/claude-sonnet-5"

# Fracción mínima de mejora (3 %) para aceptar una variante y llamar al auditor.
MOL_MIN_IMPROVEMENT: float = 0.03

# El auditor solo corre si baseline/candidato alcanza este speedup.
MOL_SPEEDUP_AUDIT_THRESHOLD: float = 1.0 + MOL_MIN_IMPROVEMENT

# Maximum correction calls per MoL variant (compilation and execution together).
MOL_FIX_ATTEMPTS: int = 3

PROMPTS_BACKENDS_DIR = PROJECT_ROOT / "prompts" / "backends"
PROMPTS_PIPELINES_DIR = PROJECT_ROOT / "prompts" / "pipelines"
PROMPTS_DIR = PROMPTS_BACKENDS_DIR  # alias de retrocompatibilidad
ENV_PATH = PROJECT_ROOT / ".env"


def resolve_model(name: str, provider: str = "openrouter") -> str:
    """Resuelve un alias corto o un ID completo al ID del proveedor indicado.

    Args:
        name: Alias corto o ID completo del modelo.
        provider: ``"openrouter"`` (por defecto) o ``"hf"`` para Hugging Face.
    """
    model_map = HF_MODELS if provider == "hf" else MODELS
    if name in model_map:
        return model_map[name]
    if name in model_map.values():
        return name
    known = ", ".join(sorted(model_map))
    provider_label = "Hugging Face" if provider == "hf" else "OpenRouter"
    raise ValueError(
        f"Unknown model for {provider_label}: {name!r}. "
        f"Use one of: {known}, or the full ID."
    )
