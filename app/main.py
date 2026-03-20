"""Main entry point — Streamlit UI for the chemistry assistant."""

import asyncio

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor import build_graph, create_initial_state

st.set_page_config(
    page_title="ChemAssistant — Synthesis Planner",
    page_icon="🧪",
    layout="wide",
)

st.title("ChemAssistant — Мультиагентный ассистент для планирования синтеза")
st.markdown("Введите целевую молекулу (название, SMILES, CAS-номер или формулу)")


# Initialize session state
if "agent_state" not in st.session_state:
    st.session_state.agent_state = create_initial_state()
    st.session_state.graph = build_graph()


def display_messages():
    """Display chat history."""
    for msg in st.session_state.agent_state["messages"]:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)


def run_agent(user_input: str):
    """Run the agent graph with user input."""
    state = st.session_state.agent_state
    state["messages"].append(HumanMessage(content=user_input))

    # Run the graph
    graph = st.session_state.graph
    result = asyncio.run(graph.ainvoke(state))

    st.session_state.agent_state = result


# Display existing messages
display_messages()

# Sidebar with state info
with st.sidebar:
    st.header("Status")
    state = st.session_state.agent_state
    st.write(f"**Phase:** {state['current_phase']}")

    if state.get("target_molecule"):
        mol = state["target_molecule"]
        st.subheader("Target Molecule")
        st.write(f"**Name:** {mol.name}")
        if mol.molecular_formula:
            st.write(f"**Formula:** {mol.molecular_formula}")
        if mol.molecular_weight:
            st.write(f"**MW:** {mol.molecular_weight:.2f} g/mol")
        if mol.image_url:
            st.image(mol.image_url, width=200)

    if state.get("synthesis_pathways"):
        st.subheader(f"Pathways: {len(state['synthesis_pathways'])}")
        if state.get("selected_pathway") is not None:
            st.write(f"**Selected:** Pathway {state['selected_pathway'] + 1}")

    if state.get("safety_report"):
        sr = state["safety_report"]
        color = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴",
        }.get(sr.overall_risk_level, "⚪")
        st.write(f"**Safety:** {color} {sr.overall_risk_level}")

    if state.get("errors"):
        st.subheader("Errors")
        for err in state["errors"]:
            st.error(f"{err.agent}: {err.message}")

    if st.button("Reset"):
        st.session_state.agent_state = create_initial_state()
        st.session_state.graph = build_graph()
        st.rerun()

# Chat input
user_input = st.chat_input("Введите молекулу или команду...")
if user_input:
    run_agent(user_input)
    st.rerun()
