"""Supervisor agent — orchestrates the multi-agent workflow via LangGraph."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.agents.calculations_agent import run_calculations
from src.agents.guard_agent import run_safety_check
from src.agents.molecule_info_agent import run_molecule_info
from src.agents.protocol_agent import run_protocol_generation
from src.agents.retrosynthesis_agent import run_retrosynthesis
from src.llm import get_llm
from src.models.state import AgentError, AgentState

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor of a multi-agent chemistry assistant.
You coordinate specialized agents to help an organic chemist plan and execute syntheses.

Your role:
1. Understand the user's request
2. Determine which agent(s) to invoke
3. Manage the workflow state
4. Present results clearly to the user

Available agents:
- molecule_info: Gets information about any chemical compound
- retrosynthesis: Plans synthesis pathways for a target molecule
- guard: Checks safety of reagents and procedures
- calculations: Computes stoichiometry and amounts
- protocol: Generates step-by-step experiment instructions

Workflow phases:
1. molecule_input → User provides target molecule
2. retrosynthesis → System finds synthesis routes
3. pathway_selection → User picks a route
4. calculations → System computes amounts
5. protocol → System generates instructions
6. done → Complete

Always explain what you're doing and present results clearly."""


def create_initial_state() -> AgentState:
    """Create a fresh initial state."""
    return {
        "messages": [],
        "target_molecule": None,
        "synthesis_pathways": [],
        "selected_pathway": None,
        "safety_report": None,
        "calculations": None,
        "protocol": None,
        "current_phase": "molecule_input",
        "errors": [],
        "next_agent": None,
    }


# --- Node functions for LangGraph ---


async def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor decides what to do next based on current state and messages."""
    llm = get_llm()

    phase = state["current_phase"]
    last_message = state["messages"][-1] if state["messages"] else None

    # Determine next action based on phase and context
    if phase == "molecule_input" and last_message:
        # User provided a molecule — run molecule info
        state["next_agent"] = "molecule_info"
        state["messages"].append(
            AIMessage(content="Looking up information about the molecule...")
        )

    elif phase == "retrosynthesis":
        state["next_agent"] = "retrosynthesis"
        state["messages"].append(
            AIMessage(content="Planning synthesis pathways...")
        )

    elif phase == "pathway_selection" and last_message:
        # Check if user selected a pathway
        content = last_message.content if hasattr(last_message, "content") else ""
        # Try to extract pathway number
        for i in range(10):
            if str(i + 1) in content or f"path_{i + 1}" in content.lower():
                state["selected_pathway"] = i
                state["current_phase"] = "calculations"
                state["next_agent"] = "calculations"
                state["messages"].append(
                    AIMessage(
                        content=f"Selected pathway {i + 1}. Calculating amounts..."
                    )
                )
                break
        else:
            state["next_agent"] = None
            state["messages"].append(
                AIMessage(
                    content="Please select a pathway by number (e.g., '1' or 'pathway 1')"
                )
            )

    elif phase == "calculations":
        state["next_agent"] = "calculations"

    elif phase == "protocol":
        state["next_agent"] = "protocol"
        state["messages"].append(
            AIMessage(content="Generating experiment protocol...")
        )

    elif phase == "done":
        state["next_agent"] = None

    else:
        state["next_agent"] = None

    return state


async def molecule_info_node(state: AgentState) -> AgentState:
    """Run MoleculeInfoAgent."""
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    try:
        molecule = await run_molecule_info(state, last_user_msg)
        state["target_molecule"] = molecule
        state["current_phase"] = "retrosynthesis"

        card = molecule.short_card()
        state["messages"].append(
            AIMessage(content=f"Found molecule:\n\n{card}\n\nStarting retrosynthesis...")
        )
    except Exception as e:
        state["errors"].append(
            AgentError(agent="molecule_info", message=str(e))
        )
        state["messages"].append(
            AIMessage(
                content=f"Error looking up molecule: {e}. Please try a different identifier."
            )
        )
        state["current_phase"] = "molecule_input"

    return state


async def retrosynthesis_node(state: AgentState) -> AgentState:
    """Run RetrosynthesisAgent."""
    target = state["target_molecule"]
    if not target:
        state["errors"].append(
            AgentError(agent="retrosynthesis", message="No target molecule set")
        )
        state["current_phase"] = "molecule_input"
        return state

    try:
        # Run retrosynthesis and safety check in parallel
        pathways = await run_retrosynthesis(state, target)
        state["synthesis_pathways"] = pathways

        # Run safety check on first pathway (parallel in production)
        if pathways and pathways[0].steps:
            safety = await run_safety_check(state, pathways[0])
            state["safety_report"] = safety

        state["current_phase"] = "pathway_selection"

        # Format pathways for display
        msg = f"Found {len(pathways)} synthesis pathway(s):\n\n"
        for i, p in enumerate(pathways):
            msg += f"**Pathway {i + 1}** ({p.total_steps} steps"
            if p.overall_yield is not None:
                msg += f", ~{p.overall_yield * 100:.0f}% overall yield"
            if p.confidence_score is not None:
                msg += f", confidence: {p.confidence_score:.0%}"
            msg += ")\n"
            for step in p.steps:
                msg += (
                    f"  Step {step.step_number}: {step.reaction_type or 'reaction'}"
                )
                if step.conditions.temperature:
                    msg += f" at {step.conditions.temperature}"
                if step.conditions.solvent:
                    msg += f" in {step.conditions.solvent}"
                if step.expected_yield:
                    msg += f" (yield ~{step.expected_yield * 100:.0f}%)"
                msg += "\n"
            msg += "\n"

        if state.get("safety_report"):
            sr = state["safety_report"]
            msg += f"\n**Safety**: Risk level = {sr.overall_risk_level}"
            if sr.has_critical_warnings():
                msg += " ⚠ CRITICAL WARNINGS"
            msg += "\n"

        msg += "\nPlease select a pathway (e.g., '1')."
        state["messages"].append(AIMessage(content=msg))

    except Exception as e:
        state["errors"].append(
            AgentError(agent="retrosynthesis", message=str(e))
        )
        state["messages"].append(
            AIMessage(content=f"Error during retrosynthesis: {e}")
        )

    return state


async def calculations_node(state: AgentState) -> AgentState:
    """Run CalculationsAgent."""
    idx = state.get("selected_pathway")
    pathways = state.get("synthesis_pathways", [])

    if idx is None or idx >= len(pathways):
        state["messages"].append(
            AIMessage(content="No pathway selected. Please select one first.")
        )
        state["current_phase"] = "pathway_selection"
        return state

    pathway = pathways[idx]

    # Default target mass — could be user-specified
    target_mass_g = 5.0  # TODO: get from user

    try:
        calculations = await run_calculations(state, pathway, target_mass_g)
        state["calculations"] = calculations
        state["current_phase"] = "protocol"

        # Display calculations
        msg = f"**Calculations for {target_mass_g} g of product:**\n\n"
        for sc in calculations.steps:
            msg += f"Step {sc.step_number}:\n"
            for r in sc.reagents:
                line = f"  - {r.name}: {r.mass_g} g ({r.moles} mol)"
                if r.volume_ml:
                    line += f" = {r.volume_ml} mL"
                msg += line + "\n"
            msg += "\n"

        msg += "Generating protocol..."
        state["messages"].append(AIMessage(content=msg))

    except Exception as e:
        state["errors"].append(
            AgentError(agent="calculations", message=str(e))
        )
        state["messages"].append(
            AIMessage(content=f"Error in calculations: {e}")
        )

    return state


async def protocol_node(state: AgentState) -> AgentState:
    """Run ProtocolAgent."""
    idx = state.get("selected_pathway", 0)
    pathways = state.get("synthesis_pathways", [])
    calculations = state.get("calculations")

    if not pathways or idx >= len(pathways) or not calculations:
        state["messages"].append(
            AIMessage(content="Missing data for protocol generation.")
        )
        return state

    pathway = pathways[idx]

    try:
        protocol = await run_protocol_generation(
            state, pathway, calculations, state.get("safety_report")
        )
        state["protocol"] = protocol
        state["current_phase"] = "done"

        # Display protocol
        msg = f"# {protocol.title}\n\n"
        msg += f"Target: {protocol.target_mass_g} g\n\n"

        if protocol.equipment_needed:
            msg += "**Equipment:** " + ", ".join(protocol.equipment_needed) + "\n\n"

        msg += "**Protocol:**\n\n"
        for step in protocol.steps:
            msg += f"**Step {step.step_number}:** {step.instruction}\n"
            if step.duration:
                msg += f"  Duration: {step.duration}\n"
            if step.safety_note:
                msg += f"  ⚠ {step.safety_note}\n"
            msg += "\n"

        if protocol.expected_yield:
            msg += f"\n**Expected yield:** {protocol.expected_yield}\n"
        if protocol.disposal_instructions:
            msg += f"\n**Disposal:** {protocol.disposal_instructions}\n"

        state["messages"].append(AIMessage(content=msg))

    except Exception as e:
        state["errors"].append(
            AgentError(agent="protocol", message=str(e))
        )
        state["messages"].append(
            AIMessage(content=f"Error generating protocol: {e}")
        )

    return state


# --- Route function ---


def route_next(state: AgentState) -> str:
    """Determine the next node based on state."""
    next_agent = state.get("next_agent")

    if next_agent == "molecule_info":
        return "molecule_info"
    elif next_agent == "retrosynthesis":
        return "retrosynthesis"
    elif next_agent == "calculations":
        return "calculations"
    elif next_agent == "protocol":
        return "protocol"
    else:
        return END


# --- Build the graph ---


def build_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("molecule_info", molecule_info_node)
    graph.add_node("retrosynthesis", retrosynthesis_node)
    graph.add_node("calculations", calculations_node)
    graph.add_node("protocol", protocol_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "molecule_info": "molecule_info",
            "retrosynthesis": "retrosynthesis",
            "calculations": "calculations",
            "protocol": "protocol",
            END: END,
        },
    )

    # All agents return to supervisor
    graph.add_edge("molecule_info", "supervisor")
    graph.add_edge("retrosynthesis", END)  # waits for user input
    graph.add_edge("calculations", "supervisor")
    graph.add_edge("protocol", END)

    return graph.compile()
