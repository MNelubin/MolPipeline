"""Research tools: intent classification, query formulation, molecule extraction.

LLM-backed steps use OpenAI-compatible API (see config.py and .env).
If OPENAI_API_KEY is unset, heuristic fallbacks are used.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from ..models.research import CandidateMolecule, ResearchQuery
from .pubchem import get_cid_by_name, get_smiles_by_cid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent classification (heuristic)
# ---------------------------------------------------------------------------

_RESEARCH_KEYWORDS_RU = [
    "хочу", "хотел бы", "ищу", "нужен", "нужна", "нужно", "нужны",
    "подбери", "подобрать", "найди", "найти", "предложи", "предложить",
    "посоветуй", "порекомендуй", "какой", "какая", "какое", "какие",
    "чем заменить", "аналог", "альтернатив",
    "ингибитор", "активатор", "антагонист", "агонист",
    "антиоксидант", "катализатор", "стабилизатор",
    "противовоспалительн", "антибактериальн", "противоопухолев",
    "анальгетик", "антибиотик", "антивирусн",
]

_RESEARCH_KEYWORDS_EN = [
    "i want", "i need", "looking for", "find me", "suggest",
    "recommend", "which", "what is a good",
    "inhibitor", "activator", "antagonist", "agonist",
    "antioxidant", "catalyst", "stabilizer",
    "anti-inflammatory", "antibacterial", "anticancer", "antitumor",
    "analgesic", "antibiotic", "antiviral",
    "drug for", "molecule for", "compound for",
]

_SMILES_CHARS = set("=()[]@/\\#%+.")
_CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def classify_user_input(text: str) -> Literal["molecule", "research_query"]:
    """Decide whether *text* is a specific molecule identifier or a research query."""
    stripped = text.strip()
    if not stripped:
        return "molecule"

    if _CAS_PATTERN.match(stripped):
        return "molecule"

    if _SMILES_CHARS & set(stripped) and " " not in stripped:
        return "molecule"

    lower = stripped.lower()

    for kw in _RESEARCH_KEYWORDS_RU:
        if kw in lower:
            return "research_query"
    for kw in _RESEARCH_KEYWORDS_EN:
        if kw in lower:
            return "research_query"

    words = lower.split()
    if len(words) >= 4:
        return "research_query"

    return "molecule"


def detect_language(text: str) -> Literal["ru", "en"]:
    for ch in text:
        if "\u0400" <= ch <= "\u04ff":
            return "ru"
    return "en"


# ---------------------------------------------------------------------------
# Heuristic fallbacks
# ---------------------------------------------------------------------------

_QUERY_TEMPLATES_EN = [
    "{keywords} molecule",
    "{keywords} drug compound",
    "{keywords} SMILES structure",
    "{keywords} pharmacology",
]

_QUERY_TEMPLATES_RU_TO_EN: dict[str, str] = {
    "ингибитор": "inhibitor", "активатор": "activator",
    "антагонист": "antagonist", "агонист": "agonist",
    "антиоксидант": "antioxidant", "катализатор": "catalyst",
    "стабилизатор": "stabilizer", "противовоспалительн": "anti-inflammatory",
    "антибактериальн": "antibacterial", "противоопухолев": "anticancer",
    "анальгетик": "analgesic", "антибиотик": "antibiotic",
    "антивирусн": "antiviral", "белок": "protein", "белка": "protein",
    "фермент": "enzyme", "фермента": "enzyme",
    "рецептор": "receptor", "рецептора": "receptor",
    "киназ": "kinase", "протеаз": "protease",
}

_STOPWORDS_RU = {
    "я", "мне", "мы", "нам", "хочу", "хотел", "бы", "ищу", "нужен",
    "нужна", "нужно", "нужны", "найди", "найти", "подбери", "подобрать",
    "предложи", "предложить", "посоветуй", "порекомендуй", "для", "от",
    "на", "по", "из", "как", "что", "чем", "какой", "какая", "какое", "какие",
}


def _formulate_search_queries_heuristic(user_input: str) -> ResearchQuery:
    lang = detect_language(user_input)
    lower = user_input.lower().strip()

    if lang == "ru":
        en_tokens: list[str] = []
        for ru_kw, en_kw in _QUERY_TEMPLATES_RU_TO_EN.items():
            if ru_kw in lower:
                en_tokens.append(en_kw)
        remaining_words = lower.split()
        remaining_words = [w for w in remaining_words if w not in _STOPWORDS_RU]
        for w in remaining_words:
            translated = False
            for ru_kw, en_kw in _QUERY_TEMPLATES_RU_TO_EN.items():
                if w.startswith(ru_kw[:4]):
                    translated = True
                    break
            if not translated and len(w) > 2:
                en_tokens.append(w)
        keywords = " ".join(dict.fromkeys(en_tokens)) if en_tokens else lower
    else:
        keywords = lower

    queries: list[str] = []
    for tmpl in _QUERY_TEMPLATES_EN:
        queries.append(tmpl.format(keywords=keywords))

    interpreted = f"Search for: {keywords}"
    return ResearchQuery(
        original_input=user_input,
        interpreted_intent=interpreted,
        search_queries=queries,
        language=lang,
    )


_DRUG_SUFFIXES = (
    "inib", "tinib", "anib", "enib", "umab", "izumab", "ximab", "lumab",
    "mab", "ine", "pine", "dine", "zine", "olol", "alol", "pril", "artan",
    "statin", "oxacin", "cillin", "vir", "navir", "previr", "nib", "lib", "sib",
    "zole", "azole", "fen", "profen", "caine", "mycin", "cycline",
    "prazole", "gliptin", "gliflozin", "tide", "mide",
)

_KNOWN_MOLECULES: set[str] = {
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
    "imatinib", "gefitinib", "erlotinib", "sunitinib", "sorafenib",
    "bortezomib", "carfilzomib",
    "trastuzumab", "bevacizumab", "rituximab", "nivolumab", "pembrolizumab",
    "metformin", "atorvastatin", "simvastatin", "omeprazole", "lansoprazole",
    "amoxicillin", "ciprofloxacin", "azithromycin", "doxycycline",
    "oseltamivir", "remdesivir", "ritonavir",
    "diazepam", "lorazepam", "fluoxetine", "sertraline",
    "morphine", "codeine", "fentanyl",
    "warfarin", "heparin", "clopidogrel",
    "insulin", "epinephrine", "dopamine", "serotonin",
    "caffeine", "nicotine", "ethanol", "methanol",
    "curcumin", "resveratrol", "quercetin", "epigallocatechin",
    "tamoxifen", "doxorubicin", "cisplatin", "paclitaxel", "vincristine",
    "celecoxib", "diclofenac", "naproxen", "indomethacin",
    "captopril", "enalapril", "losartan", "valsartan",
    "amlodipine", "nifedipine", "metoprolol", "propranolol",
    "penicillin", "vancomycin", "gentamicin", "tetracycline",
}

_SMILES_INLINE_RE = re.compile(
    r"(?<!\w)([A-Z][A-Za-z0-9@+\-\[\]\(\)\\\/=#$%.:~]{5,})(?!\w)"
)
_CAPITALIZED_WORD_RE = re.compile(r"\b([A-Z][a-z]{3,}(?:[a-z]+))\b")


def _extract_molecules_heuristic(text: str) -> list[str]:
    if not text:
        return []
    found: dict[str, None] = {}
    lower = text.lower()
    for name in _KNOWN_MOLECULES:
        if name in lower:
            found[name] = None
    for m in _CAPITALIZED_WORD_RE.finditer(text):
        word = m.group(1)
        word_lower = word.lower()
        if word_lower in _KNOWN_MOLECULES:
            found[word_lower] = None
            continue
        for suffix in _DRUG_SUFFIXES:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                found[word_lower] = None
                break
    for m in _SMILES_INLINE_RE.finditer(text):
        candidate = m.group(1)
        if any(ch in candidate for ch in "()=#[]"):
            found[candidate] = None
    return list(found)


# ---------------------------------------------------------------------------
# Public API: LLM first, then heuristic
# ---------------------------------------------------------------------------

def formulate_search_queries(user_input: str) -> ResearchQuery:
    """Translate a vague user request into search queries (LLM or heuristic)."""
    try:
        from ..services.research_llm import llm_formulate_search_queries
        rq = llm_formulate_search_queries(user_input)
        if rq is not None:
            return rq
    except Exception as exc:
        logger.warning("formulate_search_queries LLM path failed: %s", exc)
    return _formulate_search_queries_heuristic(user_input)


def extract_molecules_from_text(text: str) -> list[str]:
    """Extract molecule / drug names from text (LLM or heuristic)."""
    try:
        from ..services.research_llm import llm_extract_molecule_names
        names = llm_extract_molecule_names(text)
        if names is not None and names:
            return names
        if names is not None and not names:
            return []
    except Exception as exc:
        logger.warning("extract_molecules_from_text LLM path failed: %s", exc)
    return _extract_molecules_heuristic(text)


# ---------------------------------------------------------------------------
# Candidate resolution via PubChem
# ---------------------------------------------------------------------------

def resolve_candidates(
    names: list[str],
    original_query: str,
) -> list[CandidateMolecule]:
    """Resolve molecule names to SMILES and CIDs via PubChem."""
    candidates: list[CandidateMolecule] = []
    seen_cids: set[int] = set()

    for name in names:
        try:
            cid = get_cid_by_name(name)
            if cid is None:
                logger.debug("Could not resolve '%s' in PubChem", name)
                continue
            if cid in seen_cids:
                continue
            seen_cids.add(cid)

            smiles = get_smiles_by_cid(cid)

            candidates.append(CandidateMolecule(
                name=name,
                canonical_smiles=smiles,
                pubchem_cid=cid,
                relevance_reason=f"Found in literature related to: {original_query}",
                confidence=0.5,
                source_urls=[],
            ))
        except Exception as e:
            logger.warning("PubChem resolution failed for '%s': %s", name, e)

    return candidates
