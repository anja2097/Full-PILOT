"""Acumulador de métricas LLM compartido entre el modo single-backend y los pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

from translator.infrastructure.llm.client import fetch_generation_metadata
from translator.infrastructure.llm.models import ChatCompletion


@dataclass
class IterationMetrics:
    label: str            # e.g. "Iteración 1" o "Etapa 1/3 Vectorize — Iteración 2"
    generation_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    generation_time_ms: float | None
    cost_usd: float | None


@dataclass
class MetricsAccumulator:
    """Accumulates per-call LLM metrics across one or more translation stages."""

    calls: list[IterationMetrics] = field(default_factory=list)

    def add(self, label: str, data: ChatCompletion) -> None:
        usage = data.usage
        prompt = usage.prompt_tokens if usage else 0
        completion = usage.completion_tokens if usage else 0
        total = usage.total_tokens if usage else 0
        reasoning = (
            usage.completion_tokens_details.reasoning_tokens
            if usage and usage.completion_tokens_details
            else 0
        )
        cached = (
            usage.prompt_tokens_details.cached_tokens
            if usage and usage.prompt_tokens_details
            else 0
        )
        cost_from_usage = usage.cost if (usage and usage.cost is not None) else None
        self.calls.append(
            IterationMetrics(
                label=label,
                generation_id=data.id,
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
                reasoning_tokens=reasoning,
                cached_tokens=cached,
                generation_time_ms=None,
                cost_usd=cost_from_usage,
            )
        )

    def resolve_generation_metadata(self, api_key: str) -> None:
        for call in self.calls:
            meta = fetch_generation_metadata(api_key, call.generation_id)
            call.generation_time_ms = meta.generation_time_ms
            if meta.total_cost_usd is not None:
                call.cost_usd = meta.total_cost_usd

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_reasoning_tokens(self) -> int:
        return sum(c.reasoning_tokens for c in self.calls)

    @property
    def total_cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self.calls)

    @property
    def total_generation_time_s(self) -> float | None:
        times = [c.generation_time_ms for c in self.calls if c.generation_time_ms is not None]
        if not times:
            return None
        return sum(times) / 1000.0

    @property
    def total_cost_usd(self) -> float | None:
        costs = [c.cost_usd for c in self.calls if c.cost_usd is not None]
        if not costs:
            return None
        if len(costs) < len(self.calls):
            return None
        return sum(costs)


def print_metrics(acc: MetricsAccumulator) -> None:
    sep = "=" * 50
    print(f"\n\n{sep}")
    print("  LLM metrics (OpenRouter)")
    print(sep)

    for call in acc.calls:
        print(f"\n  -- {call.label} --")
        if call.generation_time_ms is not None:
            print(f"    Inference time:      {call.generation_time_ms / 1000:.2f} s")
        else:
            print("    Inference time:      N/A")
        if call.cost_usd is not None:
            print(f"    Cost:                ${call.cost_usd:.6f}")
        else:
            print("    Cost:                N/A")
        print(f"    Tokens prompt:       {call.prompt_tokens}")
        if call.cached_tokens:
            print(f"      (cached):          {call.cached_tokens}")
        print(f"    Tokens completion:   {call.completion_tokens}")
        if call.reasoning_tokens:
            print(f"      (reasoning):       {call.reasoning_tokens}")
        print(f"    Total tokens:        {call.total_tokens}")

    print(f"\n  -- TOTAL ({len(acc.calls)} calls) --")
    has_missing_time = any(c.generation_time_ms is None for c in acc.calls)
    gen_time = acc.total_generation_time_s
    if gen_time is not None and not has_missing_time:
        print(f"    Inference time:      {gen_time:.2f} s")
    else:
        print("    Inference time:      N/A  (not available for every iteration)")
    total_cost = acc.total_cost_usd
    if total_cost is not None:
        print(f"    Total cost:          ${total_cost:.6f}")
    else:
        has_any = any(c.cost_usd is not None for c in acc.calls)
        if has_any:
            partial = sum(c.cost_usd for c in acc.calls if c.cost_usd is not None)
            print(f"    Total cost:          N/A  (known partial: ${partial:.6f})")
        else:
            print("    Total cost:          N/A")
    print(f"    Tokens prompt:       {acc.total_prompt_tokens}")
    if acc.total_cached_tokens:
        print(f"      (cached):          {acc.total_cached_tokens}")
    print(f"    Tokens completion:   {acc.total_completion_tokens}")
    if acc.total_reasoning_tokens:
        print(f"      (reasoning):       {acc.total_reasoning_tokens}")
    print(f"    Total tokens:        {acc.total_tokens}")
    print(sep)
