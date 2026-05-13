"""Run a small verified chemistry benchmark against DeepSeek and MolPipeline.

Input CSV must contain at least:
  - level
  - prompt
  - expected_contains

The grader is intentionally simple and deterministic: a row is correct when
`expected_contains` appears in the answer after light text normalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHEM_CHAT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_MOLPIPELINE_URL = "https://hack.humaneconomy.ru"


@dataclass
class RunResult:
    row_id: str
    level: str
    system_name: str
    prompt: str
    expected_contains: str
    answer: str
    correct: bool
    elapsed_sec: float
    error: str = ""


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    replacements = {
        "–": "-",
        "—": "-",
        "−": "-",
        "‑": "-",
        "‐": "-",
        "≡": "#",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ", "")
    return text.casefold()


def is_correct(answer: str, expected_contains: str) -> bool:
    expected = normalize_text(expected_contains)
    if not expected:
        return False
    return expected in normalize_text(answer)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"level", "prompt", "expected_contains"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader, start=1):
            row = {k: (v or "").strip() for k, v in row.items()}
            row.setdefault("id", str(index))
            rows.append(row)
        return rows


def load_dotenv_file(path: Path) -> None:
    """Tiny .env loader to avoid requiring python-dotenv for benchmark runs."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def http_json_post(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib_request.Request(
        url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def ask_bare_model(prompt: str, *, max_tokens: int = 400) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    data = http_json_post(
        f"{base_url}/chat/completions",
        {
            "model": CHEM_CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a chemistry exam solver. Answer the user's chemistry multiple-choice "
                        "question concisely. Do not use external tools."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=int(os.getenv("BENCHMARK_BARE_TIMEOUT_SEC", "90")),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter response has no choices: {data}")
    return str(((choices[0].get("message") or {}).get("content") or "")).strip()


def ask_molpipeline_remote(prompt: str, *, base_url: str, timeout: int, max_tokens_hint: int = 400) -> str:
    _ = max_tokens_hint
    data = http_json_post(
        base_url.rstrip("/") + "/chat/message",
        {
            "message": prompt,
            "source_mode": "auto",
            "research_mode": "literature",
            "top_n": 5,
            "max_sources": 4,
            "client_id": "benchmark-local",
        },
        timeout=timeout,
    )
    return str(data.get("answer") or "").strip()


def run_one(row: dict[str, str], system_name: str, runner) -> RunResult:
    started = time.time()
    answer = ""
    error = ""
    try:
        answer = runner(row["prompt"])
    except Exception as exc:
        error = str(exc)
    elapsed = time.time() - started
    return RunResult(
        row_id=row.get("id") or "",
        level=row["level"],
        system_name=system_name,
        prompt=row["prompt"],
        expected_contains=row["expected_contains"],
        answer=answer,
        correct=bool(answer) and is_correct(answer, row["expected_contains"]),
        elapsed_sec=round(elapsed, 3),
        error=error,
    )


def summarize(results: list[RunResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_level": {}}
    systems = sorted({r.system_name for r in results})
    levels = sorted({r.level for r in results})

    for system in systems:
        subset = [r for r in results if r.system_name == system]
        total = len(subset)
        correct = sum(1 for r in subset if r.correct)
        summary["overall"][system] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }

    for level in levels:
        summary["by_level"][level] = {}
        for system in systems:
            subset = [r for r in results if r.level == level and r.system_name == system]
            total = len(subset)
            correct = sum(1 for r in subset if r.correct)
            summary["by_level"][level][system] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
    return summary


def write_results(results: list[RunResult], summary: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")

    with (outdir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(RunResult.__dataclass_fields__.keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    with (outdir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def _load_plotting():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for benchmark graphs: pip install matplotlib") from exc

    plt.style.use("dark_background")
    return plt


def _system_label(system_name: str) -> str:
    return {
        "bare_model": "DeepSeek",
        "molpipeline": "MolPipeline",
    }.get(system_name, system_name)


def _plot_bar_chart(title: str, rows: list[dict[str, Any]], path_base: Path) -> None:
    plt = _load_plotting()
    labels = [row["label"] for row in rows]
    values = [row["accuracy"] * 100 for row in rows]
    notes = [f"{row['correct']}/{row['total']}" for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy, %")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.0f}% ({note})" for value, note in zip(values, notes)], padding=4)

    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_by_level(summary: dict[str, Any], path_base: Path) -> None:
    plt = _load_plotting()
    levels = list(summary["by_level"])
    systems = sorted(summary["overall"])
    width = 0.36
    x_positions = list(range(len(levels)))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    for offset_index, system in enumerate(systems):
        offset = (offset_index - (len(systems) - 1) / 2) * width
        values = [
            summary["by_level"][level].get(system, {}).get("accuracy", 0.0) * 100
            for level in levels
        ]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=_system_label(system),
        )
        for bar, value, level in zip(bars, values, levels):
            stats = summary["by_level"][level].get(system, {})
            note = f"{int(stats.get('correct', 0))}/{int(stats.get('total', 0))}"
            ax.text(bar.get_x() + bar.get_width() / 2, min(value + 2, 105), note, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([level.title() for level in levels])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy, %")
    ax.set_title("Accuracy by Level: DeepSeek vs MolPipeline")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_graphs(summary: dict[str, Any], outdir: Path) -> None:
    graphs_dir = outdir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = [
        {
            "label": _system_label(system),
            "system": system,
            "accuracy": stats["accuracy"],
            "correct": stats["correct"],
            "total": stats["total"],
        }
        for system, stats in summary["overall"].items()
    ]
    _plot_bar_chart("Overall Accuracy: DeepSeek vs MolPipeline", overall_rows, graphs_dir / "overall_accuracy")

    for level, values in summary["by_level"].items():
        level_rows = [
            {
                "label": _system_label(system),
                "system": system,
                "accuracy": stats["accuracy"],
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for system, stats in values.items()
        ]
        _plot_bar_chart(
            f"{level.title()} Accuracy: DeepSeek vs MolPipeline",
            level_rows,
            graphs_dir / f"{level}_accuracy",
        )
    _plot_grouped_by_level(summary, graphs_dir / "by_level_accuracy")


def print_summary(summary: dict[str, Any], outdir: Path) -> None:
    print("Benchmark complete")
    print(f"Output: {outdir}")
    for system, stats in summary["overall"].items():
        print(f"overall {system}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")
    for level, systems in summary["by_level"].items():
        for system, stats in systems.items():
            print(f"{level} {system}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("validation_materials/chemistry_benchmark_dataset.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("validation_materials/runs/latest"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Load OpenRouter key from this .env file.")
    parser.add_argument("--molpipeline-url", default=os.getenv("MOLPIPELINE_URL", DEFAULT_MOLPIPELINE_URL))
    parser.add_argument("--molpipeline-timeout-sec", type=int, default=int(os.getenv("MOLPIPELINE_TIMEOUT_SEC", "180")))
    parser.add_argument("--limit", type=int, default=0, help="Limit number of dataset rows for smoke tests.")
    parser.add_argument("--levels", default="", help="Comma-separated level filter, for example: research or school,university.")
    parser.add_argument("--skip-bare", action="store_true")
    parser.add_argument("--skip-system", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Only validate dataset shape; do not import or call model/system.")
    args = parser.parse_args()
    load_dotenv_file(args.env_file)

    rows = load_rows(args.dataset)
    if args.levels.strip():
        wanted_levels = {level.strip() for level in args.levels.split(",") if level.strip()}
        rows = [row for row in rows if row["level"] in wanted_levels]
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.validate_only:
        print(f"Dataset valid: {len(rows)} row(s) in {args.dataset}")
        levels = sorted({row["level"] for row in rows})
        print("Levels:", ", ".join(levels))
        return 0

    runners = []
    if not args.skip_bare:
        runners.append(("bare_model", ask_bare_model))
    if not args.skip_system:
        runners.append((
            "molpipeline",
            lambda prompt: ask_molpipeline_remote(
                prompt,
                base_url=args.molpipeline_url,
                timeout=args.molpipeline_timeout_sec,
            ),
        ))
    if not runners:
        raise ValueError("Nothing to run: both --skip-bare and --skip-system were set.")

    results: list[RunResult] = []
    for row in rows:
        for system_name, runner in runners:
            print(f"[{system_name}] {row.get('id', '')} {row['level']}")
            results.append(run_one(row, system_name, runner))

    summary = summarize(results)
    write_results(results, summary, args.outdir)
    write_graphs(summary, args.outdir)
    print_summary(summary, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
