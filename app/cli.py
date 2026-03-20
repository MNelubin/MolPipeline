"""CLI interface for testing the chemistry assistant."""

import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.supervisor import build_graph, create_initial_state


async def main():
    state = create_initial_state()
    graph = build_graph()

    print("=" * 60)
    print("ChemAssistant — Synthesis Planner")
    print("Enter a molecule name, SMILES, CAS number, or formula")
    print("Type 'quit' to exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(f"[{state['current_phase']}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        state["messages"].append(HumanMessage(content=user_input))

        try:
            state = await graph.ainvoke(state)
        except Exception as e:
            print(f"\nERROR: {e}\n")
            continue

        # Print new AI messages
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                # Only print messages we haven't printed yet
                pass

        # Print the last AI message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                print(f"\n{msg.content}\n")
                break

        if state["current_phase"] == "done":
            print("\n=== Synthesis planning complete ===\n")
            break


if __name__ == "__main__":
    asyncio.run(main())
