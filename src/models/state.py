"""Agent state for LangGraph."""

from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .molecule import MoleculeInfo
from .protocol import ExperimentCalculations, ExperimentProtocol
from .reaction import SynthesisPathway
from .safety import SafetyReport


class AgentError(BaseModel):
    """An error that occurred during processing."""

    agent: str
    tool: str | None = None
    message: str
    recoverable: bool = True


class AgentState(TypedDict):
    """Shared state passed between agents in the LangGraph."""

    # Message history (LangGraph accumulator)
    messages: Annotated[list, add_messages]

    # Target molecule (filled at Step 1)
    target_molecule: MoleculeInfo | None

    # Synthesis pathways (filled at Step 2)
    synthesis_pathways: list[SynthesisPathway]

    # User's selected pathway index (filled after selection)
    selected_pathway: int | None

    # Safety report (filled in parallel with retrosynthesis)
    safety_report: SafetyReport | None

    # Stoichiometry calculations (filled at Step 4)
    calculations: ExperimentCalculations | None

    # Generated protocol (filled at Step 5)
    protocol: ExperimentProtocol | None

    # Current phase of the workflow
    current_phase: str  # molecule_input | retrosynthesis | pathway_selection |
    # calculations | protocol | done

    # Error tracking
    errors: list[AgentError]

    # Which agent should run next (set by supervisor)
    next_agent: str | None
