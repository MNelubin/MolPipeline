"""CalculationsAgent — stoichiometry and equivalents calculations."""

from src.models.protocol import ExperimentCalculations, ReagentAmount, StepCalculation
from src.models.reaction import SynthesisPathway
from src.models.state import AgentState
from src.tools.calculations import equivalents_calc, stoichiometry_calc


async def run_calculations(
    state: AgentState,
    pathway: SynthesisPathway,
    target_mass_g: float,
) -> ExperimentCalculations:
    """Calculate stoichiometry for all steps in the selected pathway.

    Works backwards from the desired product mass, accounting for
    expected yields at each step.
    """
    # Calculate how much product we need at each step,
    # working backwards from the final target
    step_targets = []
    current_mass_needed = target_mass_g

    # Reverse iterate: last step produces final product
    for step in reversed(pathway.steps):
        step_yield = step.expected_yield or 0.7  # default 70% if unknown
        # We need more input to account for yield loss
        mass_before_yield = current_mass_needed / step_yield
        step_targets.insert(0, {
            "step": step,
            "product_mass_g": current_mass_needed,
            "input_mass_needed_g": mass_before_yield,
        })
        current_mass_needed = mass_before_yield

    # Now calculate stoichiometry for each step
    step_calculations = []
    for target_info in step_targets:
        step = target_info["step"]
        product_mass = target_info["product_mass_g"]

        if step.reaction_smiles and ">>" in step.reaction_smiles:
            # Use stoichiometry_calc tool
            result = stoichiometry_calc.invoke({
                "reaction_smiles": step.reaction_smiles,
                "target_product_mass_g": product_mass,
            })

            if "error" not in result:
                reagent_amounts = []
                for r in result.get("reagents", []):
                    reagent_amounts.append(
                        ReagentAmount(
                            name=r.get("smiles", "unknown"),
                            smiles=r.get("smiles"),
                            molecular_weight=r.get("molecular_weight"),
                            equivalents=r.get("equivalents", 1.0),
                            moles=r.get("moles"),
                            mass_g=r.get("mass_g"),
                        )
                    )

                step_calculations.append(
                    StepCalculation(
                        step_number=step.step_number,
                        target_product_mass_g=product_mass,
                        target_product_moles=result.get("target_moles", 0),
                        reagents=reagent_amounts,
                        theoretical_yield_g=product_mass,
                    )
                )
            else:
                # Fallback: just record what we know
                step_calculations.append(
                    StepCalculation(
                        step_number=step.step_number,
                        target_product_mass_g=product_mass,
                        target_product_moles=0,
                        reagents=[],
                    )
                )
        else:
            step_calculations.append(
                StepCalculation(
                    step_number=step.step_number,
                    target_product_mass_g=product_mass,
                    target_product_moles=0,
                    reagents=[],
                )
            )

    # Calculate target moles for the final product
    from src.tools.rdkit_tools import rdkit_properties

    target_moles = 0.0
    if pathway.target_smiles:
        props = rdkit_properties.invoke({"smiles": pathway.target_smiles})
        if "error" not in props:
            mw = props.get("molecular_weight", 0)
            if mw > 0:
                target_moles = target_mass_g / mw

    return ExperimentCalculations(
        target_mass_g=target_mass_g,
        target_moles=round(target_moles, 6),
        steps=step_calculations,
    )
