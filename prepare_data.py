"""
Dataset preparation: loads TOUCAN, CoderForge-Preview, and OpenCodeReasoning-2,
converts each to a unified chat format, and applies weighted interleaving.

Run standalone to preview samples:
    python prepare_data.py --preview 3
"""

import argparse
import json
import random
from typing import Optional

import yaml
from datasets import Dataset, concatenate_datasets, interleave_datasets, load_dataset


# ---------------------------------------------------------------------------
# Format converters
# Each returns a list of {"role": ..., "content": ...} messages, or None to
# skip the sample.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert software engineer. Think step-by-step before acting. "
    "Use tools precisely and recover gracefully from errors."
)


def _parse_messages(value) -> Optional[list[dict]]:
    """Normalise a messages field that may be a JSON string or a native list."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value if value else None


def _format_toucan(sample: dict) -> Optional[list[dict]]:
    """
    TOUCAN schema: conversations list with role/content, tool definitions,
    and tool_call / tool_response turns.
    Passes through as-is after injecting system prompt if missing.
    """
    convs = _parse_messages(sample.get("conversations") or sample.get("messages"))
    if not convs:
        return None

    messages = []
    if not any(m.get("role") == "system" for m in convs):
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    for turn in convs:
        role = turn.get("role", "")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = turn.get("content") or turn.get("value") or ""
        msg: dict = {"role": role, "content": str(content)}
        # Preserve tool_calls on assistant turns (content may be null in OpenAI format)
        if role == "assistant" and turn.get("tool_calls"):
            msg["tool_calls"] = turn["tool_calls"]
        # Preserve tool_call_id on tool response turns
        if role == "tool" and turn.get("tool_call_id"):
            msg["tool_call_id"] = turn["tool_call_id"]
        messages.append(msg)

    return messages if len(messages) >= 2 else None


def _format_coderforge(sample: dict) -> Optional[list[dict]]:
    """
    CoderForge-Preview schema: 'messages' list with role/content turns
    representing full SWE agent trajectories (bash, file read/write, etc.).
    """
    messages = _parse_messages(sample.get("messages") or sample.get("conversations"))
    if not messages:
        return None

    out = []
    if not any(m.get("role") == "system" for m in messages):
        out.append({"role": "system", "content": SYSTEM_PROMPT})

    for turn in messages:
        role = turn.get("role", "")
        content = turn.get("content") or turn.get("value") or ""
        if role in ("system", "user", "assistant", "tool"):
            out.append({"role": role, "content": str(content)})

    return out if len(out) >= 2 else None


def _format_opencodereasoning(sample: dict) -> Optional[list[dict]]:
    """
    OpenCodeReasoning-2 schema:
      question          — the problem statement
      r1_generation     — initial solution attempt
      qwq_critique      — critique of the attempt
      solution          — final corrected solution
      judgement         — bool, whether solution is correct

    We only keep samples where judgement is True, then build a multi-turn
    chain: question → think (r1 + critique) → final solution.
    This teaches the model to reason before committing to an answer.
    """
    if not sample.get("judgement", False):
        return None

    question = sample.get("question", "").strip()
    initial = sample.get("r1_generation", "").strip()
    critique = sample.get("qwq_critique", "").strip()
    solution = sample.get("solution", "").strip()

    if not question or not solution:
        return None

    # Build a reasoning trace the model can learn from
    think_block = ""
    if initial:
        think_block += f"**Initial attempt:**\n{initial}\n\n"
    if critique:
        think_block += f"**Critique:**\n{critique}\n\n"

    if think_block:
        assistant_content = f"<think>\n{think_block.strip()}\n</think>\n\n{solution}"
    else:
        assistant_content = solution

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]


FORMATTERS = {
    "toucan": _format_toucan,
    "coderforge": _format_coderforge,
    "opencodereasoning": _format_opencodereasoning,
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_and_format(
    dataset_id: str,
    config_name: Optional[str],
    split: str,
    fmt: str,
    weight: float,
    pilot_samples: Optional[int],
    filter_language: Optional[str] = None,
) -> tuple[Dataset, float]:
    print(f"  Loading {dataset_id} ({split})...")
    ds = load_dataset(dataset_id, config_name, split=split)

    if filter_language and "language" in ds.column_names:
        ds = ds.filter(lambda x: x["language"] == filter_language, num_proc=4)
    elif filter_language and "source" in ds.column_names and fmt == "opencodereasoning":
        # OpenCodeReasoning-2 doesn't have a language column — filter is applied
        # at the dataset level by selecting the python-only subset if available,
        # otherwise we skip (the formatter already handles quality via judgement)
        pass

    if pilot_samples:
        n = min(pilot_samples, len(ds))
        ds = ds.shuffle(seed=42).select(range(n))

    formatter = FORMATTERS[fmt]

    def convert(batch):
        results = []
        for i in range(len(batch[list(batch.keys())[0]])):
            sample = {k: batch[k][i] for k in batch}
            msgs = formatter(sample)
            results.append(msgs)
        return {"messages": results}

    ds = ds.map(convert, batched=True, batch_size=1000, num_proc=4,
                remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["messages"] is not None)

    print(f"    -> {len(ds):,} samples after formatting")
    return ds, weight


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(config_path: str = "config.yaml") -> Dataset:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pilot_cfg = cfg.get("pilot", {})
    pilot_enabled = pilot_cfg.get("enabled", False)
    pilot_n = pilot_cfg.get("samples_per_dataset") if pilot_enabled else None

    if pilot_enabled:
        print(f"Pilot mode: capping each dataset at {pilot_n:,} samples")

    datasets_out = []
    weights = []

    for ds_cfg in cfg["datasets"]:
        ds, w = load_and_format(
            dataset_id=ds_cfg["id"],
            config_name=ds_cfg.get("config"),
            split=ds_cfg.get("split", "train"),
            fmt=ds_cfg["format"],
            weight=ds_cfg["weight"],
            pilot_samples=pilot_n,
            filter_language=ds_cfg.get("filter_language"),
        )
        datasets_out.append(ds)
        weights.append(w)

    # Normalize weights to probabilities
    total = sum(weights)
    probs = [w / total for w in weights]
    print(f"\nEffective mixing probabilities:")
    for ds_cfg, p in zip(cfg["datasets"], probs):
        print(f"  {ds_cfg['id']}: {p:.1%}")

    mixed = interleave_datasets(
        datasets_out,
        probabilities=probs,
        seed=42,
        stopping_strategy="all_exhausted",
    )

    mixed = mixed.shuffle(seed=42)
    print(f"\nTotal mixed dataset: {len(mixed):,} samples")
    return mixed


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=int, default=0,
                        help="Print N samples instead of building full dataset")
    parser.add_argument("--inspect", metavar="DATASET_ID",
                        help="Dump raw schema of first 2 samples from a dataset without formatting")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.inspect:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ds_cfg = next(c for c in cfg["datasets"] if args.inspect in c["id"])
        raw = load_dataset(ds_cfg["id"], ds_cfg.get("config"), split=ds_cfg.get("split", "train"))
        print(f"Columns: {raw.column_names}\n")
        for i in range(min(2, len(raw))):
            print(f"=== Row {i} ===")
            for col in raw.column_names:
                val = raw[i][col]
                snippet = str(val)[:400] + ("..." if len(str(val)) > 400 else "")
                print(f"  [{col}]: {snippet}")
            print()
        import sys; sys.exit(0)

    ds = build_dataset(args.config)

    if args.preview:
        print(f"\n--- Previewing {args.preview} samples ---\n")
        for i in random.sample(range(len(ds)), min(args.preview, len(ds))):
            print(f"=== Sample {i} ===")
            for msg in ds[i]["messages"]:
                role = msg["role"].upper()
                content = msg["content"][:300] + ("..." if len(msg["content"]) > 300 else "")
                print(f"[{role}] {content}\n")
            print()
