from __future__ import annotations

from collections import deque
from typing import Callable

from mcp_server.models import Pictogram

# Sentinel concept index for pictograms added via synset expansion
# (not directly tied to any planner concept).
_MAX_CONCEPT = 9999


#################################################################################################
# Strategy: sequential_blocks
#################################################################################################

def _sequential_blocks(
    candidates: list[Pictogram],
    selected_ids: set[int],
    concept_order: dict[int, int],
) -> list[Pictogram]:
    """Baseline strategy: one sorted block per concept, by AAC quality score desc.

    Eval baseline, must remain bit-for-bit identical to the pre-refactor
    implementation in agent.py.
    """
    def _sort_key(pic: Pictogram) -> tuple[int, int]:
        cidx  = concept_order.get(pic.id, _MAX_CONCEPT)
        # Higher score = higher quality: aac_color > aac > no-violence > no-sex
        score = 0
        if pic.aac_color:    score += 4
        if pic.aac:          score += 2
        if not pic.violence: score += 1
        if not pic.sex:      score += 1
        return (cidx, -score)  # ascending concept index, descending score

    pool = [p for p in candidates if p.id not in selected_ids]
    pool.sort(key=_sort_key)
    return pool


#################################################################################################
# Strategy: round_robin_weighted
#################################################################################################

# weight(rank) = DECAY ** rank  (rank = position in ascending concept_order)
_ROUND_ROBIN_DECAY = 0.7


def _round_robin_weighted(
    candidates: list[Pictogram],
    selected_ids: set[int],
    concept_order: dict[int, int],
) -> list[Pictogram]:
    """Weighted interleaving across concept groups (deficit round-robin).

    Each concept forms a FIFO queue. At every step the group with the highest
    deficit is picked:
    
        deficit(c) = weight(c) * (total_picks + 1) - picks(c)
    
    Weights decrease with concept position so earlier concepts get more
    representation without starving later ones.

    Weights are assigned by rank in the sorted list of present concept indices,
    not by raw index value, so non-contiguous indices are handled correctly. 
    Synset-expanded candidates (cidx == _MAX_CONCEPT) always land in the 
    lowest-weight group.
    """
    pool_candidates = [p for p in candidates if p.id not in selected_ids]

    # Group pictograms by their concept index
    groups: dict[int, list[Pictogram]] = {}
    for p in pool_candidates:
        cidx = concept_order.get(p.id, _MAX_CONCEPT)
        groups.setdefault(cidx, []).append(p)

    concept_indices = sorted(groups.keys())
    if not concept_indices:
        return []

    # Assign exponentially decaying weights by rank, not raw index
    weights     = {cidx: _ROUND_ROBIN_DECAY ** rank for rank, cidx in enumerate(concept_indices)}
    queues      = {cidx: deque(groups[cidx]) for cidx in concept_indices}
    picks       = {cidx: 0 for cidx in concept_indices}
    total_picks = 0

    pool: list[Pictogram] = []
    while any(queues[c] for c in concept_indices):
        ready  = [c for c in concept_indices if queues[c]]
        # Pick the group with the highest weighted deficit
        chosen = max(ready, key=lambda c: weights[c] * (total_picks + 1) - picks[c])
        pool.append(queues[chosen].popleft())
        picks[chosen] += 1
        total_picks   += 1

    return pool


#################################################################################################
# Registry + public entry point
#################################################################################################

STRATEGIES: dict[str, Callable[[list[Pictogram], set[int], dict[int, int]], list[Pictogram]]] = {
    "sequential_blocks":    _sequential_blocks,
    "round_robin_weighted": _round_robin_weighted,
}


def rank_and_fill(
    candidates: list[Pictogram],
    selected_ids: set[int],
    concept_order: dict[int, int],
    max_results: int,
    strategy: str = "sequential_blocks",
) -> tuple[list[Pictogram], list[int]]:
    """Sort candidates with *strategy* and truncate to max_results.

    Returns (window, pool_ids) where pool_ids is the full pre-truncation
    order used by AACAgent to populate last_pool_ids for eval.
    Raises ValueError for unknown strategy names.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown ranking strategy: {strategy!r}. Available: {sorted(STRATEGIES)}"
        )

    pool     = STRATEGIES[strategy](candidates, selected_ids, concept_order)
    pool_ids = [p.id for p in pool]
    window   = pool[:max_results]
    return window, pool_ids
