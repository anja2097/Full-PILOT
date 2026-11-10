"""Punto de entrada CLI del traductor."""

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path

from translator.config.env import load_env
from translator.config.settings import (
    API_URL,
    CHECKSUM_RTOL,
    DEFAULT_BACKEND_NAME,
    DEFAULT_MODEL,
    ENV_PATH,
    HF_API_URL,
    HF_MODELS,
    MOL_DEFAULT_AGENT_MODELS,
    MOL_DEFAULT_AUDIT_MODEL,
    MODELS,
    THINKING_EFFORTS,
    resolve_model,
)
from translator.exceptions import PipelineStageError, TranslatorError
from translator.registry import discover_backends, discover_pipelines, resolve_backend, resolve_pipeline
from translator.services.pipeline_runner import run_pipeline
from translator.services.translate import translate_file

logger = logging.getLogger(__name__)


def parse_compile_flags(raw: str) -> list[str]:
    """Convierte la cadena de --flags en tokens de compilación."""
    return shlex.split(raw) if raw.strip() else []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate serial C/C++ to a parallel backend using an LLM via OpenRouter.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Source .c or .cpp file to translate",
    )

    # ── Modo de traducción: backend único o pipeline multi-etapa ─────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "-b",
        "--backend",
        default=None,
        metavar="NAME",
        help=(
            f"Single-backend parallelization target (default: {DEFAULT_BACKEND_NAME}). "
            "Incompatible with --pipeline."
        ),
    )
    mode.add_argument(
        "--pipeline",
        default=None,
        metavar="PIPELINE",
        help=(
            "Multi-stage pipeline, e.g. Vectorize-OpenMP-MPI. "
            "Incompatible with -b/--backend."
        ),
    )

    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Show the available backends and exit",
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="Show the pipelines available in prompts/pipelines/ and exit",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        metavar="ALIAS",
        help=f"Model to use (default: {DEFAULT_MODEL}). Accepts a short alias or a full OpenRouter ID.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Show the available models and exit",
    )
    parser.add_argument(
        "--model-provider",
        metavar="PROVIDER",
        help=(
            "Force a specific inference provider via provider.only "
            "(OpenRouter only; e.g. Together)"
        ),
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        action="store_true",
        help="Enable extended reasoning in the API",
    )
    thinking.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable extended reasoning (default)",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=THINKING_EFFORTS,
        metavar="LEVEL",
        help="Reasoning effort level (requires --thinking)",
    )
    parser.add_argument(
        "--flags",
        type=str,
        default="",
        metavar="FLAGS",
        help=(
            "Extra flags before the source file, as one quoted string "
            '(e.g. "-DPOLYBENCH_TIME utilities/polybench.c")'
        ),
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        metavar="TOL",
        help=(
            "Checksum tolerance against the serial reference "
            f"(default {CHECKSUM_RTOL:g}: relative, or absolute when the reference is 0). "
            "Without a checksum, comparison stays an exact stdout match."
        ),
    )
    parser.add_argument(
        "--compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Verify correctness in a loop (default): if the translated program's checksum "
            "differs from the serial reference (or the program fails at runtime), retry "
            "with execution-error fixes until MAX_RETRIES is exhausted. "
            "--no-compare disables it."
        ),
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Use the Hugging Face Inference API instead of OpenRouter",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print the request and response JSON of every API call "
            "(translation, fixes, MoL agents and the auditor) without interleaving them"
        ),
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help=(
            "Print token breakdown, inference time and USD cost "
            "accumulated across LLM calls. OpenRouter only."
        ),
    )

    # ── MoL and optimisation-stage flags (pipeline mode only) ─────────────
    parser.add_argument(
        "--mol-models",
        nargs="+",
        metavar="MODEL",
        help=(
            "Models for the MoL generator agents in optimization stages "
            f"(default: {', '.join(MOL_DEFAULT_AGENT_MODELS)}). "
            "Accepts short aliases or full OpenRouter IDs. Valid only with --pipeline."
        ),
    )
    parser.add_argument(
        "--mol-audit-model",
        metavar="MODEL",
        help=(
            f"Model for the MoL auditor (default: {MOL_DEFAULT_AUDIT_MODEL}). "
            "Valid only with --pipeline."
        ),
    )
    return parser


def _print_backends() -> None:
    backends = discover_backends()
    print("Available backends:\n")
    for backend in sorted(backends.values(), key=lambda b: b.name):
        default = " (default)" if backend.name == DEFAULT_BACKEND_NAME else ""
        print(f"  {backend.name:<14} [{backend.slug}]  prompts: {backend.prompts_dir}{default}")
    print()


def _print_pipelines() -> None:
    pipelines = discover_pipelines()
    if not pipelines:
        print("No pipelines available in prompts/pipelines/.\n")
        return
    print("Available pipelines:\n")
    for name in pipelines:
        stages = name.split("-")
        print(f"  {name:<36} ({' → '.join(stages)})")
    print()


def _print_models(hf: bool = False) -> None:
    if hf:
        print("Available models (Hugging Face):\n")
        for alias, model_id in sorted(HF_MODELS.items()):
            print(f"  {alias:<20} → {model_id}")
        print("\nYou can also pass a full Hugging Face ID with -m/--model.")
    else:
        print("Available models (OpenRouter):\n")
        for alias, model_id in sorted(MODELS.items()):
            default = " (default)" if alias == DEFAULT_MODEL else ""
            print(f"  {alias:<20} → {model_id}{default}")
        print("\nYou can also pass a full OpenRouter ID with -m/--model.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )
    if args.debug:
        logging.getLogger("translator.infrastructure.llm").setLevel(logging.DEBUG)

    if args.list_backends:
        _print_backends()
        return

    if args.list_pipelines:
        _print_pipelines()
        return

    if args.list_models:
        _print_models(hf=args.hf)
        return

    if not args.source:
        parser.error(
            "a source file path is required "
            "(or use --list-backends / --list-pipelines / --list-models)"
        )

    if args.thinking_effort and not args.thinking:
        parser.error("--thinking-effort requires --thinking")

    if args.metrics and args.hf:
        parser.error("--metrics is only available with OpenRouter (do not use it with --hf)")

    if args.model_provider and args.hf:
        parser.error("--model-provider is only available with OpenRouter (do not use it with --hf)")

    if args.pipeline and args.hf:
        parser.error("--pipeline is only available with OpenRouter (do not use it with --hf)")

    if args.mol_models and not args.pipeline:
        parser.error("--mol-models is only available in --pipeline mode")
    if args.mol_audit_model and not args.pipeline:
        parser.error("--mol-audit-model is only available in --pipeline mode")
    if (args.mol_models or args.mol_audit_model) and args.hf:
        parser.error("--mol-models / --mol-audit-model are not available with --hf")

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"[ERROR] File not found: {source_path}")
        sys.exit(1)
    if source_path.suffix not in (".c", ".cpp"):
        print("[ERROR] Only .c or .cpp files are accepted")
        sys.exit(1)

    provider = "hf" if args.hf else "openrouter"
    try:
        model_id = resolve_model(args.model, provider=provider)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print("Use --list-models to see the available aliases.")
        sys.exit(1)

    load_env(ENV_PATH)
    if args.hf:
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            print("[ERROR] HF_API_KEY not found in .env (required with --hf)")
            sys.exit(1)
        api_url = HF_API_URL
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("[ERROR] OPENROUTER_API_KEY not found in .env")
            sys.exit(1)
        api_url = API_URL

    thinking = args.thinking
    if thinking and args.hf:
        logger.warning(
            "[WARNING] --thinking is not supported by the Hugging Face Inference API. "
            "The reasoning flag will be ignored."
        )
        thinking = False

    extra_flags = parse_compile_flags(args.flags)
    rtol = CHECKSUM_RTOL if args.rtol is None else args.rtol
    provider_label = "Hugging Face" if args.hf else "OpenRouter"

    common_log_kwargs = dict(
        model_id=model_id,
        provider_label=provider_label,
        model_provider=args.model_provider,
        thinking=thinking,
        thinking_effort=args.thinking_effort,
        extra_flags=extra_flags,
        compare=args.compare,
        debug=args.debug,
        metrics=args.metrics,
    )

    if args.pipeline:
        # ── Pipeline mode ─────────────────────────────────────────────────────
        try:
            pipeline = resolve_pipeline(args.pipeline)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            print("Use --list-pipelines to see the available pipelines.")
            sys.exit(1)

        # Resolve MoL models (aliases or full IDs)
        mol_agent_models: list[str] | None = None
        if args.mol_models:
            mol_agent_models = []
            for m in args.mol_models:
                try:
                    mol_agent_models.append(resolve_model(m, provider="openrouter"))
                except ValueError:
                    # Accept full IDs that are not in the alias map
                    mol_agent_models.append(m)

        mol_audit_model: str | None = None
        if args.mol_audit_model:
            try:
                mol_audit_model = resolve_model(args.mol_audit_model, provider="openrouter")
            except ValueError:
                mol_audit_model = args.mol_audit_model

        logger.info("\n\nPipeline mode: %s → %s", source_path.name, pipeline.name)
        logger.info("  Provider:   %s", provider_label)
        logger.info("  Model:      %s", model_id)
        if args.model_provider:
            logger.info("  LLM prov.:  %s (forced)", args.model_provider)
        if thinking:
            effort = args.thinking_effort or "medium (default)"
            logger.info("  Reasoning:  enabled (effort: %s)", effort)
        else:
            logger.info("  Reasoning:  disabled")
        if extra_flags:
            logger.info("  Compile flags: %s", " ".join(extra_flags))
        logger.info("  Compare:    %s", "enabled" if args.compare else "disabled")
        if mol_agent_models:
            logger.info("  MoL agents: %s", ", ".join(mol_agent_models))
        if mol_audit_model:
            logger.info("  MoL auditor: %s", mol_audit_model)
        if args.debug:
            logger.info("  Debug:      enabled (JSON of every request and response)")
        if args.metrics:
            logger.info("  Metrics:    enabled")
        logger.info("\n")

        try:
            run_pipeline(
                source_path,
                api_key,
                pipeline=pipeline,
                model=model_id,
                api_url=api_url,
                model_provider=args.model_provider,
                thinking=thinking,
                thinking_effort=args.thinking_effort,
                extra_flags=extra_flags,
                compare=args.compare,
                metrics=args.metrics,
                mol_agent_models=mol_agent_models,
                mol_audit_model=mol_audit_model,
                debug=args.debug,
                rtol=rtol,
            )
        except PipelineStageError as exc:
            print(
                f"[ERROR] Pipeline {pipeline.name!r} failed at stage "
                f"{exc.stage_index + 1} ({exc.stage_name}): {exc}"
            )
            sys.exit(1)
        except TranslatorError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

    else:
        # ── Single-backend mode ───────────────────────────────────────────────
        backend_name = args.backend or DEFAULT_BACKEND_NAME
        try:
            backend = resolve_backend(backend_name)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            print("Use --list-backends to see the available backends.")
            sys.exit(1)

        translated_name = backend.translated_path(source_path).name
        logger.info("\n\nTranslating %s -> %s", source_path.name, translated_name)
        logger.info("  Backend:    %s", backend.name)
        logger.info("  Provider:   %s", provider_label)
        logger.info("  Model:      %s", model_id)
        if args.model_provider:
            logger.info("  LLM prov.:  %s (forced)", args.model_provider)
        if thinking:
            effort = args.thinking_effort or "medium (default)"
            logger.info("  Reasoning:  enabled (effort: %s)", effort)
        else:
            logger.info("  Reasoning:  disabled")
        if extra_flags:
            logger.info("  Compile flags: %s", " ".join(extra_flags))
        logger.info(
            "  Compare:    %s",
            "enabled (checksum verification in a loop)" if args.compare else "disabled",
        )
        if args.debug:
            logger.info("  Debug:      enabled (JSON of every request and response)")
        if args.metrics:
            logger.info("  Metrics:    enabled (tokens, time and USD cost per iteration)")
        logger.info("\n")

        try:
            translate_file(
                source_path,
                api_key,
                backend=backend,
                model=model_id,
                api_url=api_url,
                model_provider=args.model_provider,
                thinking=thinking,
                thinking_effort=args.thinking_effort,
                extra_flags=extra_flags,
                compare=args.compare,
                metrics=args.metrics,
                rtol=rtol,
            )
        except TranslatorError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
