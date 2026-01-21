import json
from typing import Dict, List, Optional
import re


def build_openfactscore_input(
    entity: Dict,
    drop_empty: bool = False,
    deduplicate: bool = True,
) -> Optional[Dict]:
    """
    Convert one extracted-claims entity into an OpenFactScore-compatible sample.

    Parameters
    ----------
    entity : dict
        One JSON object from your extracted claims file.
    drop_empty : bool
        Whether to drop entities with no valid atomic facts. --> Note that if this is True, the assert in ofs_eval will fail because the function will skip these entries and the length of the returned results will be < 2500
    deduplicate : bool
        Whether to remove duplicate atomic facts (preserving order).

    Returns
    -------
    dict or None
        OpenFactScore sample, or None if dropped.
    """

    # --- topic ---
    # prompt_source = [id, topic]
    try:
        topic = entity["prompt_source"][1]
    except Exception:
        raise ValueError(f"Invalid prompt_source format: {entity.get('prompt_source')}")

    # --- generation ---
    generation = entity.get("response", "").strip()

    # --- atomic facts ---
    raw_claims = entity.get("all_claims", [])

    atomic_facts: List[str] = []
    seen = set()

    for claim in raw_claims:
        if claim is None:
            continue
        claim = claim.strip()
        if not claim:
            continue

        if deduplicate:
            if claim in seen:
                continue
            seen.add(claim)

        atomic_facts.append(claim)

    if drop_empty and len(atomic_facts) == 0:
        return None

    return {
        "topic": topic,
        "generation": generation,
        "atomic_facts": atomic_facts,

    }

def get_knowledge_source_name(entity_idx: int, source: str, id:str):
    """
    Returns the naming convention used for storing the knowledge source for a given entity given:
    
    :param entity_idx: the index of the entity in the HF data.
    :type entity: int
    :param source: the original source of the entity: either "doi" for scholalry paper or "iep" for the Internet Encyclopedia of Philosophy
    :type source: str
    :param id: the id of the entity depending on the source, either its doi (for scholalry papers) or title (for iep)
    :type id: str
    """
    if source == "doi":
        doi_id = id.strip().replace("/", "_")
        return str(entity_idx)+"_"+doi_id
    
    elif source == "iep":
        iep_id = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", id.strip().lower()))
        return  str(entity_idx)+"_"+iep_id
    
    else:
        raise ValueError("Source for knowledge source name must be either 'doi' or 'iep'")
    
