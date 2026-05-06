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

    session_id = chem_chat_store.ensure_session(
        None,
        "Find aspirin synthesis route",
        "auto",
        client_id="client-a",
    )
    chem_chat_store.append_message(session_id, "user", "Find aspirin synthesis route")
    chem_chat_store.append_message(
        session_id,
        "assistant",
        "Route found.",
        {"result": {"status": "ok"}},
    )

    sessions = chem_chat_store.list_sessions(client_id="client-a")
    assert sessions[0]["id"] == session_id
    assert sessions[0]["client_id"] == "client-a"
    assert sessions[0]["message_count"] == 2
    assert sessions[0]["last_message"] == "Route found."

    detail = chem_chat_store.get_session(session_id, client_id="client-a")
    assert detail is not None
    assert detail["client_id"] == "client-a"
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["payload"]["result"]["status"] == "ok"

    assert chem_chat_store.delete_session(session_id, client_id="client-a") is True
    assert chem_chat_store.get_session(session_id, client_id="client-a") is None


def test_store_scopes_sessions_by_client_id(monkeypatch, tmp_path):
    _use_temp_store(monkeypatch, tmp_path)

    first_id = chem_chat_store.ensure_session(None, "first private chat", "auto", client_id="client-a")
    second_id = chem_chat_store.ensure_session(None, "second private chat", "auto", client_id="client-b")
    chem_chat_store.append_message(first_id, "user", "visible only to client a")
    chem_chat_store.append_message(second_id, "user", "visible only to client b")

    assert [item["id"] for item in chem_chat_store.list_sessions(client_id="client-a")] == [first_id]
    assert [item["id"] for item in chem_chat_store.list_sessions(client_id="client-b")] == [second_id]
    assert chem_chat_store.get_session(first_id, client_id="client-b") is None
    assert chem_chat_store.delete_session(first_id, client_id="client-b") is False
    assert chem_chat_store.get_session(first_id, client_id="client-a") is not None


def test_chat_request_persists_user_and_assistant_messages(monkeypatch, tmp_path):
    _use_temp_store(monkeypatch, tmp_path)
    result_payload = {
        "status": "ok",
        "intent": "general",
        "model": "deepseek/deepseek-v4-flash",
        "plan": {"tools": []},
        "answer": "Saved answer.",
        "tools_used": [],
        "artifacts": {},
        "suggested_next_actions": [],
    }

    with patch("mvp.api.run_chem_chat", return_value=result_payload):
        response = asyncio.run(_run_chem_chat_request(ChemChatRequest(
            client_id="client-test",
            message="what is chemistry?",
        )))

    assert response.session_id
    detail = chem_chat_store.get_session(response.session_id, client_id="client-test")
    assert detail is not None
    assert [message["role"] for message in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "what is chemistry?"
    assert detail["messages"][1]["payload"]["result"]["answer"] == "Saved answer."
