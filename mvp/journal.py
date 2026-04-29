"""Agent journal — write-only side.

Usage in a node:
    from .journal import AgentJournal

    def my_node(state):
        j = AgentJournal.for_session(state.get("session_id", "default"))
        with j.step("my_node", phase=1):
            j.decision("my_node", "Выбран маршрут #2", {"route": 2}, phase=1)
            return {...}

No imports from the rest of mvp — only stdlib.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

EventType = Literal["start", "end", "tool_call", "decision", "warning", "error"]
StatusType = Literal["ok", "warning", "error", "running"]

LOGS_DIR = Path(__file__).parent.parent / "logs"

_NODE_PHASE: dict[str, int] = {
    "classify": 1,
    "validate_and_guard": 1,
    "research": 1,
    "research_fallback": 1,
    "molecule_info": 1,
    "interrupt_card": 1,
    "retrosynthesis": 2,
    "guard_safety": 2,
    "reagent_check": 2,
    "aggregate": 2,
    "interrupt_select_pathway": 2,
    "stoichiometry": 3,
    "experiment_planner": 3,
}

_NODE_LABEL: dict[str, str] = {
    "classify": "Классификатор запроса",
    "validate_and_guard": "Валидация + Гвард",
    "research": "Агент поиска (research)",
    "research_fallback": "Повторный поиск",
    "molecule_info": "Агент свойств молекулы",
    "interrupt_card": "Ожидание подтверждения (Фаза 1)",
    "retrosynthesis": "Агент ретросинтеза",
    "guard_safety": "Проверка безопасности",
    "reagent_check": "Проверка реагентов",
    "aggregate": "Агрегатор маршрутов",
    "interrupt_select_pathway": "Ожидание выбора маршрута (Фаза 2)",
    "stoichiometry": "Стехиометрия",
    "experiment_planner": "Протокол эксперимента",
}


class AgentJournal:
    """Thread-safe append-only journal for one MAS session."""

    _instances: dict[str, "AgentJournal"] = {}
    _lock: threading.Lock = threading.Lock()

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = LOGS_DIR / f"{session_id}.jsonl"
        self._file_lock = threading.Lock()

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def for_session(cls, session_id: str) -> "AgentJournal":
        """Return (or create) the journal for session_id."""
        with cls._lock:
            if session_id not in cls._instances:
                cls._instances[session_id] = cls(session_id)
            return cls._instances[session_id]

    @classmethod
    def close_session(cls, session_id: str) -> None:
        with cls._lock:
            cls._instances.pop(session_id, None)

    # ── Core write ────────────────────────────────────────────────────────────

    def log(
        self,
        node: str,
        event: EventType,
        summary: str,
        data: dict[str, Any] | None = None,
        *,
        phase: int | None = None,
        status: StatusType = "ok",
        duration_ms: int | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "phase": phase if phase is not None else _NODE_PHASE.get(node, 0),
            "node": node,
            "node_label": _NODE_LABEL.get(node, node),
            "event": event,
            "status": status,
            "summary": summary,
        }
        if data:
            record["data"] = data
        if duration_ms is not None:
            record["duration_ms"] = duration_ms

        line = json.dumps(record, ensure_ascii=False)
        with self._file_lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def step(self, node: str, phase: int | None = None):
        t0 = time.monotonic()
        self.log(node, "start", f"[{node}] начало работы", phase=phase, status="running")
        try:
            yield self
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            self.log(
                node, "error", f"[{node}] ошибка: {exc}",
                {"error": str(exc), "type": type(exc).__name__},
                phase=phase, status="error", duration_ms=elapsed,
            )
            raise
        else:
            elapsed = int((time.monotonic() - t0) * 1000)
            self.log(node, "end", f"[{node}] завершено", phase=phase, status="ok", duration_ms=elapsed)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def tool_call(
        self,
        node: str,
        tool_name: str,
        args: dict[str, Any] | None = None,
        result_summary: str = "",
        phase: int | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.log(
            node, "tool_call",
            f"Инструмент: {tool_name} → {result_summary}",
            {"tool": tool_name, "args": args or {}},
            phase=phase, duration_ms=duration_ms,
        )

    def decision(
        self,
        node: str,
        summary: str,
        data: dict[str, Any] | None = None,
        phase: int | None = None,
    ) -> None:
        self.log(node, "decision", summary, data, phase=phase)

    def warning(
        self,
        node: str,
        summary: str,
        data: dict[str, Any] | None = None,
        phase: int | None = None,
    ) -> None:
        self.log(node, "warning", summary, data, phase=phase, status="warning")

    # ── Read helpers ──────────────────────────────────────────────────────────

    def read_events(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        events = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events

    @staticmethod
    def list_sessions() -> list[str]:
        if not LOGS_DIR.exists():
            return []
        return [
            p.stem
            for p in sorted(LOGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        ]

    @property
    def path(self) -> Path:
        return self._path

    # ── Markdown export ───────────────────────────────────────────────────────

    def export_markdown(self, output_path: Path | None = None) -> Path:
        events = self.read_events()
        md_path = output_path or self._path.with_suffix(".md")

        _PHASE_NAMES = {
            1: "Фаза 1: Идентификация молекулы",
            2: "Фаза 2: Ретросинтез",
            3: "Фаза 3: Протокол эксперимента",
        }
        _ICONS = {
            "start": "▶", "end": "✓", "tool_call": "🔧",
            "decision": "💡", "warning": "⚠️", "error": "✗",
        }

        def _fmt_ts(ts_str: str) -> str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                return dt.strftime("%H:%M:%S")
            except Exception:
                return ts_str

        def _fmt_date(ts_str: str) -> str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                return dt.strftime("%d.%m.%Y")
            except Exception:
                return ""

        lines: list[str] = []
        first_ts = events[0].get("ts", "") if events else ""
        date_str = _fmt_date(first_ts)
        lines.append(f"# Журнал МАС — сессия `{self.session_id}`")
        if date_str:
            lines.append(f"> {date_str}")
        lines.append("")

        total = len(events)
        warnings = sum(1 for e in events if e.get("event") == "warning")
        errors = sum(1 for e in events if e.get("status") == "error")
        phases = sorted({e.get("phase", 0) for e in events if e.get("phase")})
        lines.append(
            f"**Событий:** {total} &nbsp;|&nbsp; "
            f"**Предупреждений:** {warnings} &nbsp;|&nbsp; "
            f"**Ошибок:** {errors} &nbsp;|&nbsp; "
            f"**Фаз:** {', '.join(str(p) for p in phases)}"
        )
        lines.append("")

        if len(events) >= 2:
            try:
                t0_dt = datetime.fromisoformat(events[0]["ts"].replace("Z", "+00:00"))
                t1_dt = datetime.fromisoformat(events[-1]["ts"].replace("Z", "+00:00"))
                elapsed_s = int((t1_dt - t0_dt).total_seconds())
                lines.append(f"**Общее время:** {elapsed_s} с")
                lines.append("")
            except Exception:
                pass

        lines.append("---")
        lines.append("")

        current_phase: int | None = None
        for ev in events:
            phase = ev.get("phase", 0)
            event_type = ev.get("event", "")
            node_label = ev.get("node_label", ev.get("node", ""))
            summary = ev.get("summary", "")
            ts = _fmt_ts(ev.get("ts", ""))
            duration_ms = ev.get("duration_ms")
            data = ev.get("data")

            if phase != current_phase:
                current_phase = phase
                phase_name = _PHASE_NAMES.get(phase, f"Фаза {phase}")
                lines.append(f"## {phase_name}")
                lines.append("")

            if event_type == "start":
                continue

            icon = _ICONS.get(event_type, "•")
            duration_str = f" _({duration_ms} мс)_" if duration_ms is not None else ""

            lines.append(f"{icon} `{ts}` **{node_label}**{duration_str}")
            if summary:
                clean = summary
                if event_type == "end" and "завершено" in clean and not data:
                    clean = "завершено"
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;{clean}")

            if data and event_type != "start":
                important: list[str] = []
                for k, v in data.items():
                    if k in ("tool", "args", "type"):
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        important.append(f"`{k}`: {v}")
                    elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                        important.append(f"`{k}`: {', '.join(v[:5])}")
                if important:
                    lines.append("")
                    lines.append("&nbsp;&nbsp;&nbsp;&nbsp;" + " &nbsp;·&nbsp; ".join(important))

            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path
