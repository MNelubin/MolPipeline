"""SQLite persistence for ChemChat sessions and messages."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DATA_DIR

_DB_PATH = DATA_DIR / "chem_chat_sessions.db"
_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{8,120}$")
_DEFAULT_CLIENT_ID = "default"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    _init_db()
    conn = sqlite3.connect(str(_DB_PATH), timeout=30.0)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), timeout=30.0)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL DEFAULT 'default',
                    title TEXT NOT NULL,
                    source_mode TEXT NOT NULL DEFAULT 'auto',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }
            if "client_id" not in columns:
                conn.execute(
                    "ALTER TABLE chat_sessions "
                    "ADD COLUMN client_id TEXT NOT NULL DEFAULT 'default'"
                )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    payload_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages(session_id, id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_client_updated
                ON chat_sessions(client_id, updated_at DESC)
            """)
            conn.commit()
        finally:
            conn.close()
        _INITIALIZED = True


def _safe_session_id(session_id: str | None = None) -> str:
    if session_id and _SESSION_ID_RE.match(session_id):
        return session_id
    return f"chat-{uuid.uuid4().hex[:16]}"


def _safe_client_id(client_id: str | None = None) -> str:
    value = (client_id or "").strip()
    if value and _CLIENT_ID_RE.match(value):
        return value
    return _DEFAULT_CLIENT_ID


def _make_title(message: str) -> str:
    title = " ".join(message.strip().split())
    if not title:
        return "Новый чат"
    return title[:72] + ("..." if len(title) > 72 else "")


def ensure_session(
    session_id: str | None,
    first_message: str,
    source_mode: str = "auto",
    client_id: str | None = None,
) -> str:
    safe_id = _safe_session_id(session_id)
    safe_client_id = _safe_client_id(client_id)
    now = _utc_now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, client_id FROM chat_sessions WHERE id = ?",
            (safe_id,),
        ).fetchone()
        if row is not None and row["client_id"] != safe_client_id:
            safe_id = _safe_session_id(None)
            row = None
        if row is None:
            conn.execute(
                """
                INSERT INTO chat_sessions (
                    id, client_id, title, source_mode, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    safe_id,
                    safe_client_id,
                    _make_title(first_message),
                    source_mode or "auto",
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?, source_mode = ?
                WHERE id = ? AND client_id = ?
                """,
                (now, source_mode or "auto", safe_id, safe_client_id),
            )
        conn.commit()
    return safe_id


def append_message(
    session_id: str,
    role: str,
    content: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    now = _utc_now()
    payload_json = json.dumps(payload or {}, ensure_ascii=False)
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, payload_json, now),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        conn.commit()
        message_id = cursor.lastrowid
    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "payload": payload or {},
        "created_at": now,
    }


def _short_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _compact_tool_memory(payload: dict[str, Any]) -> str:
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return ""
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        return ""

    lines: list[str] = []
    tools = result.get("tools_used") or []
    if tools:
        lines.append(f"tools_used: {', '.join(str(tool) for tool in tools[:8])}")

    molecule = artifacts.get("molecule") or {}
    if molecule:
        validation = molecule.get("validation") or {}
        label = molecule.get("query_used") or molecule.get("query")
        parts = [
            f"label={_short_text(label, 80)}" if label else "",
            f"smiles={molecule.get('smiles')}" if molecule.get("smiles") else "",
            f"formula={validation.get('molecular_formula')}" if validation.get("molecular_formula") else "",
            f"cid={molecule.get('pubchem_cid')}" if molecule.get("pubchem_cid") else "",
        ]
        lines.append("molecule: " + "; ".join(part for part in parts if part))

    safety = artifacts.get("safety") or {}
    if safety:
        taxonomy = safety.get("safety_taxonomy") or {}
        reason = (
            (((taxonomy.get("blocked_categories") or taxonomy.get("warning_categories") or [{}])[0]).get("reason"))
            or (safety.get("explosive_check") or {}).get("reason")
            or (safety.get("molecule_check") or {}).get("reason")
            or (safety.get("reaction_check") or {}).get("reason")
        )
        lines.append(
            "safety: "
            f"overall={safety.get('overall_status') or 'UNKNOWN'}"
            + (f"; reason={_short_text(reason, 180)}" if reason else "")
        )
        categories = [
            f"{item.get('hazard_type')}:{item.get('status')}/{item.get('danger_level')}"
            for item in (taxonomy.get("categories") or [])[:6]
        ]
        if categories:
            lines.append(f"safety_taxonomy: {', '.join(categories)}")

    retro = artifacts.get("retrosynthesis") or {}
    routes = retro.get("routes") or []
    if retro:
        lines.append(
            "retrosynthesis: "
            f"depth_mode={retro.get('depth_mode')}; "
            f"total_found={retro.get('total_found')}; "
            f"total_unique={retro.get('total_unique')}; "
            f"sources={', '.join(str(source) for source in (retro.get('sources_used') or [])[:6])}"
        )
        tree = retro.get("multi_step_tree") or {}
        if tree:
            lines.append(f"multi_step_tree: stats={_short_text(tree.get('stats'), 360)}")
        for index, route in enumerate(routes[:3], start=1):
            lines.append(
                f"route_{index}: "
                f"source={route.get('source_label') or route.get('source')}; "
                f"score={route.get('final_score')}; "
                f"reactants={_short_text(route.get('reactants'), 360)}"
            )

    availability = artifacts.get("availability") or {}
    items = availability.get("items") or []
    if availability:
        lines.append(f"availability: summary={_short_text(availability.get('summary'), 360)}")
        for index, item in enumerate(items[:8], start=1):
            lines.append(
                f"availability_item_{index}: "
                f"label={_short_text(item.get('label') or item.get('input'), 120)}; "
                f"smiles={item.get('smiles') or item.get('canonical_smiles')}; "
                f"available={item.get('available')}; "
                f"source={item.get('source_label') or item.get('source')}; "
                f"ppg={item.get('ppg')}"
            )

    admet = artifacts.get("admet") or {}
    if admet:
        overall = admet.get("overall") or {}
        lines.append(
            "admet: "
            f"score={overall.get('score')}; "
            f"risk={overall.get('risk_level')}; "
            f"safety_overlay={_short_text(admet.get('safety_overlay'), 260)}"
        )

    research = artifacts.get("research") or {}
    if research:
        lines.append(f"research: summary={_short_text(research.get('summary'), 420)}")
        for index, source in enumerate((research.get("sources") or [])[:6], start=1):
            lines.append(
                f"source_{index}: "
                f"title={_short_text(source.get('title') or source.get('name'), 160)}; "
                f"url={source.get('url')}"
            )

    if not lines:
        return ""
    return "[Tool memory from previous assistant turn]\n" + "\n".join(lines[:32])


def list_sessions(limit: int = 50, client_id: str | None = None) -> list[dict[str, Any]]:
    safe_client_id = _safe_client_id(client_id)
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.client_id,
                s.title,
                s.source_mode,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count,
                (
                    SELECT content
                    FROM chat_messages lm
                    WHERE lm.session_id = s.id
                    ORDER BY lm.id DESC
                    LIMIT 1
                ) AS last_message
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.client_id = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (safe_client_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_session(session_id: str, client_id: str | None = None) -> dict[str, Any] | None:
    if not _SESSION_ID_RE.match(session_id):
        return None
    safe_client_id = _safe_client_id(client_id)
    with _connect() as conn:
        session = conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ? AND client_id = ?",
            (session_id, safe_client_id),
        ).fetchone()
        if session is None:
            return None
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, payload_json, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    messages: list[dict[str, Any]] = []
    for row in rows:
        payload = {}
        if row["payload_json"]:
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                payload = {}
        messages.append({
            "id": row["id"],
            "session_id": row["session_id"],
            "role": row["role"],
            "content": row["content"],
            "payload": payload,
            "created_at": row["created_at"],
        })
    result = dict(session)
    result["messages"] = messages
    return result


def get_context_messages(
    session_id: str,
    limit: int = 8,
    client_id: str | None = None,
) -> list[dict[str, str]]:
    session = get_session(session_id, client_id)
    if not session:
        return []
    messages = session.get("messages", [])
    context: list[dict[str, str]] = []
    for item in messages:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not content:
            continue
        text = str(content)
        if role == "assistant":
            memory = _compact_tool_memory(item.get("payload") or {})
            if memory:
                text = f"{text}\n\n{memory}"
        context.append({"role": role, "content": text[:6000]})
    return context[-limit:]


def delete_session(session_id: str, client_id: str | None = None) -> bool:
    if not _SESSION_ID_RE.match(session_id):
        return False
    safe_client_id = _safe_client_id(client_id)
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM chat_sessions WHERE id = ? AND client_id = ?",
            (session_id, safe_client_id),
        )
        conn.commit()
        return cursor.rowcount > 0
