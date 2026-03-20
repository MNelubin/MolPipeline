"""Main entry point — Streamlit UI for the chemistry assistant."""

import asyncio

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor import build_graph, create_initial_state

st.set_page_config(
    page_title="ChemAssistant",
    page_icon="🧪",
    layout="wide",
)

# --- CSS ---
st.markdown("""
<style>
.phase-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
}
.phase-molecule_input { background: #e3f2fd; color: #1565c0; }
.phase-retrosynthesis { background: #fff3e0; color: #e65100; }
.phase-pathway_selection { background: #f3e5f5; color: #7b1fa2; }
.phase-calculations { background: #e8f5e9; color: #2e7d32; }
.phase-protocol { background: #fce4ec; color: #c62828; }
.phase-done { background: #e0f2f1; color: #00695c; }
.source-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 0.75em;
    margin-right: 4px;
}
.source-ibm_rxn { background: #0062ff; color: white; }
.source-askcos { background: #6a1b9a; color: white; }
.source-ord { background: #00796b; color: white; }
.source-llm_fallback { background: #757575; color: white; }
.source-predicted { background: #757575; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("ChemAssistant")
st.caption("Multi-agent synthesis planner: IBM RXN + ASKCOS + PubChem + RDKit + LLM")


# Initialize session state
if "agent_state" not in st.session_state:
    st.session_state.agent_state = create_initial_state()
    st.session_state.graph = build_graph()


PHASE_LABELS = {
    "molecule_input": "Ввод молекулы",
    "retrosynthesis": "Ретросинтез",
    "pathway_selection": "Выбор пути",
    "calculations": "Расчёты",
    "protocol": "Протокол",
    "done": "Готово",
}


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

    graph = st.session_state.graph
    result = asyncio.run(graph.ainvoke(state))

    st.session_state.agent_state = result


# Display existing messages
display_messages()


# --- Sidebar ---
with st.sidebar:
    state = st.session_state.agent_state
    phase = state["current_phase"]

    # Phase indicator
    phase_label = PHASE_LABELS.get(phase, phase)
    st.markdown(
        f'<span class="phase-badge phase-{phase}">{phase_label}</span>',
        unsafe_allow_html=True,
    )

    # Target molecule card
    if state.get("target_molecule"):
        mol = state["target_molecule"]
        st.divider()
        st.subheader(mol.name or "Target")

        if mol.image_url:
            st.image(mol.image_url, width=200)

        cols = st.columns(2)
        if mol.molecular_formula:
            cols[0].metric("Formula", mol.molecular_formula)
        if mol.molecular_weight:
            cols[1].metric("MW", f"{mol.molecular_weight}")

        if mol.smiles:
            st.code(mol.smiles, language=None)

    # Synthesis pathways
    if state.get("synthesis_pathways"):
        st.divider()
        pathways = state["synthesis_pathways"]
        st.subheader(f"Pathways ({len(pathways)})")

        for i, p in enumerate(pathways):
            source = p.steps[0].source if p.steps else "?"
            selected = state.get("selected_pathway") == i
            prefix = "-> " if selected else ""

            with st.expander(
                f"{prefix}Path {i+1}: {p.total_steps} steps",
                expanded=selected,
            ):
                st.markdown(
                    f'<span class="source-badge source-{source}">{source}</span>',
                    unsafe_allow_html=True,
                )
                if p.overall_yield is not None:
                    st.write(f"Yield: ~{p.overall_yield * 100:.0f}%")
                if p.confidence_score is not None:
                    st.progress(
                        min(p.confidence_score, 1.0),
                        text=f"Confidence: {p.confidence_score:.0%}",
                    )
                for step in p.steps:
                    st.write(
                        f"**Step {step.step_number}:** "
                        f"{step.reaction_type or 'reaction'}"
                    )

    # Safety report
    if state.get("safety_report"):
        st.divider()
        sr = state["safety_report"]
        risk_colors = {
            "low": "🟢", "medium": "🟡",
            "high": "🟠", "critical": "🔴",
        }
        icon = risk_colors.get(sr.overall_risk_level, "⚪")
        st.subheader(f"Safety {icon}")
        st.write(f"Risk: **{sr.overall_risk_level}**")
        if sr.requires_fume_hood:
            st.warning("Fume hood required")
        if sr.requires_inert_atmosphere:
            st.warning("Inert atmosphere required")
        if sr.required_ppe:
            st.write("PPE: " + ", ".join(sr.required_ppe))

    # Errors
    if state.get("errors"):
        st.divider()
        for err in state["errors"]:
            st.error(f"[{err.agent}] {err.message[:100]}")

    # Reset
    st.divider()
    if st.button("Reset", use_container_width=True):
        st.session_state.agent_state = create_initial_state()
        st.session_state.graph = build_graph()
        st.rerun()

# --- Chat input ---
placeholder = {
    "molecule_input": "Enter molecule (name, SMILES, CAS, formula)...",
    "pathway_selection": "Select pathway number (1, 2, 3...)",
    "done": "Start new synthesis (type molecule name)...",
}.get(phase, "Enter command...")

user_input = st.chat_input(placeholder)
if user_input:
    if phase == "done":
        # Reset for new molecule
        st.session_state.agent_state = create_initial_state()
        st.session_state.graph = build_graph()
    run_agent(user_input)
    st.rerun()
