"""Tests for persisted ChemChat sessions."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from .. import chem_chat_store
from ..api import ChemChatRequest, _run_chem_chat_request


def _use_temp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(chem_chat_store, "_DB_PATH", tmp_path / "chem_chat_sessions.db")
    monkeypatch.setattr(chem_chat_store, "_INITIALIZED", False)


def test_store_creates_lists_loads_and_deletes_session(monkeypatch, tmp_path):
    _use_temp_store(monkeypatch, tmp_path)

    session_id = chem_chat_store.ensure_session(None, "Найди путь синтеза аспирина", "auto")
    chem_chat_store.append_message(session_id, "user", "Найди путь синтеза аспирина")
    chem_chat_store.append_message(session_id, "assistant", "Маршрут найден.", {"result": {"status": "ok"}})

    sessions = chem_chat_store.list_sessions()
    assert sessions[0]["id"] == session_id
    assert sessions[0]["message_count"] == 2
    assert sessions[0]["last_message"] == "Маршрут найден."

    detail = chem_chat_store.get_session(session_id)
    assert detail is not None
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["payload"]["result"]["status"] == "ok"

    assert chem_chat_store.delete_session(session_id) is True
    assert chem_chat_store.get_session(session_id) is None


def test_chat_request_persists_user_and_assistant_messages(monkeypatch, tmp_path):
    _use_temp_store(monkeypatch, tmp_path)
    result_payload = {
        "status": "ok",
        "intent": "general",
        "model": "deepseek/deepseek-v4-flash",
        "plan": {"tools": []},
        "answer": "Ответ сохранен.",
        "tools_used": [],
        "artifacts": {},
        "suggested_next_actions": [],
    }

    with patch("mvp.api.run_chem_chat", return_value=result_payload):
        response = asyncio.run(_run_chem_chat_request(ChemChatRequest(message="что такое химия?")))

    assert response.session_id
    detail = chem_chat_store.get_session(response.session_id)
    assert detail is not None
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "что такое химия?"
    assert detail["messages"][1]["payload"]["result"]["answer"] == "Ответ сохранен."
