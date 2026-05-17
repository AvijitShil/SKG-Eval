
import numpy as np
import re as _re
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from skg.graph_manager import get_edges_by_turn


NEGATION_MARKERS = frozenset({
    "not", "no", "never", "doesn't", "does not", "cannot",
    "isn't", "is not", "aren't", "are not", "won't", "will not",
    "doesn't", "does not", "didn't", "did not", "has no", "have no"
})


ANTONYM_RELATION_PAIRS: frozenset = frozenset({

    frozenset({"increases", "decreases"}),
    frozenset({"increases", "reduces"}),
    frozenset({"rises", "falls"}),
    frozenset({"rises", "drops"}),
    frozenset({"grows", "shrinks"}),
    frozenset({"expands", "contracts"}),
    frozenset({"improves", "worsens"}),
    frozenset({"improves", "degrades"}),
    frozenset({"strengthens", "weakens"}),
    frozenset({"accelerates", "slows"}),
    frozenset({"speeds up", "slows down"}),
    frozenset({"raises", "lowers"}),
    frozenset({"gains", "loses"}),
    frozenset({"adds", "removes"}),
    frozenset({"higher", "lower"}),
    frozenset({"more", "less"}),

    frozenset({"causes", "prevents"}),
    frozenset({"causes", "avoids"}),
    frozenset({"leads to", "prevents"}),
    frozenset({"results in", "prevents"}),
    frozenset({"produces", "eliminates"}),
    frozenset({"triggers", "prevents"}),
    frozenset({"induces", "inhibits"}),
    frozenset({"promotes", "inhibits"}),
    frozenset({"encourages", "discourages"}),
    frozenset({"enables", "prevents"}),
    frozenset({"facilitates", "prevents"}),

    frozenset({"helps", "hurts"}),
    frozenset({"helps", "harms"}),
    frozenset({"benefits", "harms"}),
    frozenset({"supports", "opposes"}),
    frozenset({"supports", "contradicts"}),
    frozenset({"treats", "causes"}),
    frozenset({"cures", "causes"}),
    frozenset({"heals", "damages"}),

    frozenset({"confirms", "denies"}),
    frozenset({"proves", "disproves"}),
    frozenset({"supports", "refutes"}),
    frozenset({"validates", "invalidates"}),
    frozenset({"agrees", "disagrees"}),
    frozenset({"accepts", "rejects"}),
    frozenset({"allows", "prohibits"}),
    frozenset({"permits", "forbids"}),

    frozenset({"always", "never"}),
    frozenset({"always", "sometimes"}),
    frozenset({"always", "rarely"}),
    frozenset({"always", "occasionally"}),
    frozenset({"always", "seldom"}),
    frozenset({"always", "can"}),
    frozenset({"never", "sometimes"}),
    frozenset({"never", "often"}),
    frozenset({"never", "can"}),
    frozenset({"generally", "never"}),
    frozenset({"typically", "never"}),

    frozenset({"converges", "diverges"}),
    frozenset({"stabilizes", "destabilizes"}),
    frozenset({"reduces", "increases"}),
    frozenset({"minimizes", "maximizes"}),
    frozenset({"overfits", "underfits"}),
    frozenset({"helps reduce", "increases"}),
    frozenset({"reduces", "raises"}),

    frozenset({"includes", "excludes"}),
    frozenset({"contains", "lacks"}),
    frozenset({"requires", "prevents"}),
    frozenset({"depends on", "independent of"}),
    frozenset({"is same as", "is different from"}),
    frozenset({"is identical to", "is different from"}),
    frozenset({"equals", "differs from"}),
})


_COMPARABLE_ATTRIBUTES = frozenset({
    "definition", "effect", "property", "comparison",
    "requirement", "quantity", "negation"
})


NODE_SIM_THRESHOLD = 0.60
REL_SIM_THRESHOLD  = 0.30
OBJ_DIV_THRESHOLD  = 0.25


SAME_TYPE_PATH_B_THRESHOLD = 0.85


SAME_TYPE_MIN_CONFIDENCE = 0.60


ELABORATION_OBJ_FLOOR = 0.20


_SAME_TYPE_GROUPS: list = [
    frozenset({"person"}),
    frozenset({"organization"}),
    frozenset({"event"}),
    frozenset({"number"}),
    frozenset({"time"}),
    frozenset({"object"}),
    frozenset({"condition"}),
    frozenset({"concept"}),

    frozenset({"person", "organization"}),
    frozenset({"number", "object"}),
    frozenset({"time", "number"}),
]


_SKIP_NODES = {
    "you", "i", "we", "it", "they", "people", "person", "someone",
    "thing", "something", "that", "this", "a lot", "key", "issue",
    "answer", "what", "which", "there", "here", "one", "all",
    "most", "many", "some", "both", "each", "any",
}


def _has_negation_flip(rel_a: str, rel_b: str) -> bool:
    def is_negated(rel: str) -> bool:
        rel_lower = rel.lower()
        for marker in NEGATION_MARKERS:
            if marker in rel_lower:
                return True
        return False

    a_neg = is_negated(rel_a)
    b_neg = is_negated(rel_b)


    return a_neg != b_neg


def _is_antonym_relation_pair(rel_a: str, rel_b: str) -> bool:
    rel_a_lower = rel_a.lower()
    rel_b_lower = rel_b.lower()
    words_a = set(rel_a_lower.split())
    words_b = set(rel_b_lower.split())

    for pair in ANTONYM_RELATION_PAIRS:
        pair_list = list(pair)
        word_0 = pair_list[0]
        word_1 = pair_list[1]
        is_multi_0 = len(word_0.split()) > 1
        is_multi_1 = len(word_1.split()) > 1


        a_has_0 = (word_0 in rel_a_lower) if is_multi_0 else (word_0 in words_a)
        a_has_1 = (word_1 in rel_a_lower) if is_multi_1 else (word_1 in words_a)
        b_has_0 = (word_0 in rel_b_lower) if is_multi_0 else (word_0 in words_b)
        b_has_1 = (word_1 in rel_b_lower) if is_multi_1 else (word_1 in words_b)


        if (a_has_0 and b_has_1) or (a_has_1 and b_has_0):
            return True

    return False


def _get_node_type(graph, node_key: str) -> str:
    if node_key not in graph:
        return ""
    return str(graph.nodes[node_key].get("type", "")).lower().strip()


def _are_same_type_group(type_a: str, type_b: str) -> bool:
    if not type_a or not type_b:
        return False
    for group in _SAME_TYPE_GROUPS:
        if type_a in group and type_b in group:
            return True
    return False


def _extract_number(s: str):
    m = _re.search(r'\b(\d+(?:\.\d+)?)\b', s)
    return float(m.group(1)) if m else None


def _check_node_geometric(
    graph,
    node_key: str,
    current_turn_id: int,
    embedding_model,
    geom_min_rel_sim: float = 0.15,
    geom_min_obj_sim: float = 0.10,
    geom_rel_diverge: float = 0.35,
    geom_obj_diverge: float = 0.20,
) -> tuple[bool, float, str]:
    curr_edges, hist_edges = get_edges_by_turn(graph, node_key, current_turn_id)

    if not curr_edges or not hist_edges:
        return False, 0.0, "no_edges"

    best_is_contra  = False
    best_confidence = 0.0
    best_reason     = "none"

    curr_rels = [e["rel"] for e in curr_edges]
    hist_rels = [e["rel"] for e in hist_edges]
    curr_objs = [e["obj"] for e in curr_edges]
    hist_objs = [e["obj"] for e in hist_edges]

    curr_rel_embs = embedding_model.encode(curr_rels)
    hist_rel_embs = embedding_model.encode(hist_rels)
    curr_obj_embs = embedding_model.encode(curr_objs)
    hist_obj_embs = embedding_model.encode(hist_objs)

    if curr_rel_embs.ndim == 1: curr_rel_embs = curr_rel_embs.reshape(1, -1)
    if hist_rel_embs.ndim == 1: hist_rel_embs = hist_rel_embs.reshape(1, -1)
    if curr_obj_embs.ndim == 1: curr_obj_embs = curr_obj_embs.reshape(1, -1)
    if hist_obj_embs.ndim == 1: hist_obj_embs = hist_obj_embs.reshape(1, -1)

    rel_sim_matrix = cos_sim(curr_rel_embs, hist_rel_embs)
    obj_sim_matrix = cos_sim(curr_obj_embs, hist_obj_embs)

    _SUPPRESS_TYPES = ("elaboration", "solution", "diagnosis")

    for ci, curr_e in enumerate(curr_edges):
        for hi, hist_e in enumerate(hist_edges):

            rel_sim       = float(rel_sim_matrix[ci][hi])
            obj_sim       = float(obj_sim_matrix[ci][hi])
            curr_rel      = curr_e["rel"]
            hist_rel      = hist_e["rel"]
            curr_rel_type = curr_e.get("rel_type", "assertion")
            hist_rel_type = hist_e.get("rel_type", "assertion")
            curr_intent   = curr_e.get("intent", "STATE")
            hist_intent   = hist_e.get("intent", "STATE")
            curr_prop_type = curr_e.get("property_type", "ADDITIVE")
            hist_prop_type = hist_e.get("property_type", "ADDITIVE")


            if _has_negation_flip(curr_rel, hist_rel):
                if hist_rel_type in _SUPPRESS_TYPES:
                    print(
                        f"    [GEOM] Negation flip suppressed: "
                        f"hist='{hist_rel}' [{hist_rel_type}] -- additive pattern"
                    )
                    continue
                if curr_rel_type in _SUPPRESS_TYPES:
                    print(
                        f"    [GEOM] Negation flip suppressed: "
                        f"curr='{curr_rel}' [{curr_rel_type}] -- additive pattern"
                    )
                    continue

                if obj_sim > 0.40:
                    confidence = 0.95
                    reason = (
                        f"NEGATION_FLIP | "
                        f"curr='{curr_rel}->{curr_e['obj']}' [{curr_rel_type}] "
                        f"hist='{hist_rel}->{hist_e['obj']}' [{hist_rel_type}]"
                    )
                    print(
                        f"    [GEOM] Negation flip: '{hist_rel}' vs '{curr_rel}' "
                        f"-> CONTRADICTION (conf={confidence})"
                    )
                    if confidence > best_confidence:
                        best_is_contra  = True
                        best_confidence = confidence
                        best_reason     = reason
                else:
                    print(f"    [GEOM] Negation flip suppressed: obj_sim={obj_sim:.3f} < 0.40. Claims are independent.")
                continue


            if (curr_rel_type not in _SUPPRESS_TYPES
                    and hist_rel_type not in _SUPPRESS_TYPES
                    and _is_antonym_relation_pair(curr_rel, hist_rel)):

                if obj_sim > geom_min_obj_sim:
                    confidence = 0.88
                    reason = (
                        f"ANTONYM_PAIR | rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} | "
                        f"curr='{curr_rel}->{curr_e['obj']}' [{curr_rel_type}] "
                        f"hist='{hist_rel}->{hist_e['obj']}' [{hist_rel_type}]"
                    )
                    print(
                        f"    [GEOM] Antonym pair: '{hist_rel}' vs '{curr_rel}' "
                        f"obj_sim={obj_sim:.3f} -> CONTRADICTION (conf={confidence})"
                    )
                    if confidence > best_confidence:
                        best_is_contra  = True
                        best_confidence = confidence
                        best_reason     = reason
                else:
                    print(
                        f"    [GEOM] Antonym pair suppressed (unrelated objects): "
                        f"'{curr_rel}' vs '{hist_rel}' | obj_sim={obj_sim:.3f} "
                        f"< {geom_min_obj_sim} -> different properties"
                    )
                continue


            if curr_intent != hist_intent:
                print(
                    f"    [GEOM] Intent gate: curr={curr_intent} vs "
                    f"hist={hist_intent} -- different modalities, skipping"
                )
                continue


            if curr_rel_type in _SUPPRESS_TYPES:
                continue


            if rel_sim < geom_min_rel_sim or obj_sim < geom_min_obj_sim:
                print(
                    f"    [GEOM] SKIP (noise floor): "
                    f"rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} | "
                    f"curr='{curr_rel}' hist='{hist_rel}'"
                )
                continue


            if rel_sim > 0.70:
                curr_num = _extract_number(curr_e["obj"])
                hist_num = _extract_number(hist_e["obj"])
                if (curr_num is not None
                        and hist_num is not None
                        and curr_num != hist_num):
                    confidence = 0.92
                    reason = (
                        f"NUMERIC_MISMATCH | "
                        f"curr_num={curr_num} hist_num={hist_num} | "
                        f"rel_sim={rel_sim:.3f} | "
                        f"curr='{curr_e['rel']}->{curr_e['obj']}' "
                        f"hist='{hist_e['rel']}->{hist_e['obj']}'"
                    )
                    print(
                        f"    [GEOM] Numeric mismatch: {hist_num} -> {curr_num} "
                        f"| rel_sim={rel_sim:.3f} -> CONTRADICTION (conf={confidence})"
                    )
                    if confidence > best_confidence:
                        best_is_contra  = True
                        best_confidence = confidence
                        best_reason     = reason
                    continue


            if rel_sim > 0.85:


                if curr_prop_type != "EXCLUSIVE" or hist_prop_type != "EXCLUSIVE":
                    print(
                        f"    [GEOM] PATH_B Suppressed (Mutual Exclusivity Failed): "
                        f"curr='{curr_rel}->{curr_e['obj']}' [{curr_prop_type}], "
                        f"hist='{hist_rel}->{hist_e['obj']}' [{hist_prop_type}]. "
                        f"At least one claim is ADDITIVE, so they safely coexist."
                    )
                    continue

                curr_obj_type = _get_node_type(graph, curr_e["obj"])
                hist_obj_type = _get_node_type(graph, hist_e["obj"])
                same_type     = _are_same_type_group(curr_obj_type, hist_obj_type)


                if curr_e["obj"].lower().strip() == hist_e["obj"].lower().strip():
                    print(
                        f"    [GEOM] PATH_B skip: objects are identical "
                        f"('{curr_e['obj']}') — reinforcement [{curr_prop_type}]"
                    )
                    continue


                if obj_sim < 0.35:
                    confidence = 1.0 - obj_sim
                    reason = (
                        f"PATH_B_STANDARD(obj_diverge) | "
                        f"rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} | "
                        f"curr='{curr_rel}->{curr_e['obj']}' [Type:{curr_obj_type}|Prop:{curr_prop_type}] "
                        f"hist='{hist_rel}->{hist_e['obj']}' [Type:{hist_obj_type}|Prop:{hist_prop_type}]"
                    )
                    print(
                        f"    [GEOM] PATH_B standard: rel_sim={rel_sim:.3f} "
                        f"obj_sim={obj_sim:.3f}<0.35 "
                        f"\u2192 conf={confidence:.3f} (Property: {curr_prop_type})"
                    )
                    if confidence > best_confidence:
                        best_is_contra  = True
                        best_confidence = confidence
                        best_reason     = reason


                elif same_type and obj_sim < SAME_TYPE_PATH_B_THRESHOLD:
                    confidence = max(SAME_TYPE_MIN_CONFIDENCE, 1.0 - obj_sim)
                    reason = (
                        f"PATH_B_SAMETYPE(obj_diverge) | "
                        f"rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} | "
                        f"type='{curr_obj_type}'='{hist_obj_type}' | "
                        f"curr='{curr_rel}->{curr_e['obj']}' [{curr_prop_type}] "
                        f"hist='{hist_rel}->{hist_e['obj']}' [{hist_prop_type}]"
                    )
                    print(
                        f"    [GEOM] PATH_B same-type: rel_sim={rel_sim:.3f} "
                        f"obj_sim={obj_sim:.3f}<{SAME_TYPE_PATH_B_THRESHOLD} "
                        f"type={curr_obj_type} "
                        f"\u2192 conf={confidence:.3f} (Property: {curr_prop_type})"
                    )
                    if confidence > best_confidence:
                        best_is_contra  = True
                        best_confidence = confidence
                        best_reason     = reason


            if rel_sim > 0.5:
                if obj_sim < 0.4:

                    confidence = 1.0 - obj_sim
                    reason = (
                        f"SEMANTIC_DRIFT (strong) | "
                        f"rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} | "
                        f"curr='{curr_rel}->{curr_e['obj']}' hist='{hist_rel}->{hist_e['obj']}'"
                    )
                    print(f"    [GEOM] SEMANTIC_DRIFT (strong): obj_sim={obj_sim:.3f} < 0.4 -> conf={confidence:.3f}")
                    if confidence > best_confidence:
                        best_is_contra  = True
                        best_confidence = confidence
                        best_reason     = reason

                elif obj_sim < 0.75:

                    confidence = 0.450
                    reason = (
                        f"SEMANTIC_DRIFT (moderate) | "
                        f"rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} | "
                        f"curr='{curr_rel}->{curr_e['obj']}' hist='{hist_rel}->{hist_e['obj']}'"
                    )
                    print(f"    [GEOM] SEMANTIC_DRIFT (moderate): rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} -> conf={confidence:.3f}")
                    if confidence > best_confidence:
                        best_is_contra  = True
                        best_confidence = confidence
                        best_reason     = reason
                else:
                    print(f"    [GEOM] Semantic match, no contradiction: rel_sim={rel_sim:.3f} obj_sim={obj_sim:.3f} -> compatible claims (similar relations, similar objects)")

    return best_is_contra, best_confidence, best_reason


def _get_current_turn_id(graph, new_nodes: list[str]):
    for node_key in new_nodes:
        if node_key in graph:
            turn_id = graph.nodes[node_key].get("turn_id")
            if turn_id is not None:
                return turn_id

    max_tid = None
    for _, _, data in graph.edges(data=True):
        tid = data.get("turn_id")
        if tid is not None:
            if max_tid is None or tid > max_tid:
                max_tid = tid
    return max_tid


def check_relational_contradiction(
    graph,
    new_nodes: list[str],
    triples: list[dict],
    embedding_model,
    turn_history: list = None,
    current_turn_text: str = "",
    geom_min_rel_sim: float = 0.15,
    geom_min_obj_sim: float = 0.10,
    geom_rel_diverge: float = 0.35,
    geom_obj_diverge: float = 0.20,
) -> float:

    current_turn_id = _get_current_turn_id(graph, new_nodes)

    if current_turn_id is None:
        print("    [GEOM] No current turn_id found — skipping contradiction check.")
        return 1.0

    if not new_nodes and not triples:
        return 1.0


    BLOCKLIST = {
        "it", "they", "this", "that", "these", "those",
        "a lot", "something", "anything", "everything", "nothing",
        "people", "person", "someone", "everyone",
        "thing", "things", "our", "their",
    }


    candidate_nodes = set(new_nodes)


    for u, _, data in graph.edges(data=True):
        if data.get("turn_id") == current_turn_id and data.get("kind") == "fact":
            candidate_nodes.add(u)

    print(f"    [GEOM] Candidates: {len(candidate_nodes)} nodes for contradiction check (turn_id={current_turn_id})")

    max_contradiction_confidence = 0.0
    contradiction_found = False

    for node_key in sorted(candidate_nodes):


        if node_key.lower().strip() in BLOCKLIST:
            print(f"    [GEOM] '{node_key}' in blocklist — skipping")
            continue

        is_contra, confidence, reason = _check_node_geometric(
            graph=graph,
            node_key=node_key,
            current_turn_id=current_turn_id,
            embedding_model=embedding_model,
            geom_min_rel_sim=geom_min_rel_sim,
            geom_min_obj_sim=geom_min_obj_sim,
            geom_rel_diverge=geom_rel_diverge,
            geom_obj_diverge=geom_obj_diverge,
        )

        if is_contra:
            contradiction_found = True
            print(
                f"    [GEOM] *** CONTRADICTION DETECTED *** "
                f"node='{node_key}' | {reason}"
            )
            if confidence > max_contradiction_confidence:
                max_contradiction_confidence = confidence

    if not contradiction_found:
        print(f"    [GEOM] No contradictions found.")
        return 1.0


    s_logic = max(0.0, 1.0 - max_contradiction_confidence)
    print(f"    [GEOM] S_logic = {s_logic:.3f} (max_confidence={max_contradiction_confidence:.3f})")
    return float(s_logic)


def _get_near_twins(
    graph,
    node_key: str,
    current_turn_nodes: set,
) -> list[tuple[str, float]]:
    current_turn = _get_node_turn(graph, node_key)
    seen = set()
    twins = []

    for _, neighbor, data in graph.edges(node_key, data=True):
        if data.get("kind") != "semantic" or neighbor in seen:
            continue
        if neighbor in current_turn_nodes:
            continue
        sim = data.get("weight", 0.0)
        if sim < NODE_SIM_THRESHOLD:
            continue
        if graph.nodes[neighbor].get("quarantined", False):
            continue
        if _get_node_turn(graph, neighbor) >= current_turn:
            continue
        twins.append((neighbor, sim))
        seen.add(neighbor)

    for source, _, data in graph.in_edges(node_key, data=True):
        if data.get("kind") != "semantic" or source in seen:
            continue
        if source in current_turn_nodes:
            continue
        sim = data.get("weight", 0.0)
        if sim < NODE_SIM_THRESHOLD:
            continue
        if graph.nodes[source].get("quarantined", False):
            continue
        if _get_node_turn(graph, source) >= current_turn:
            continue
        twins.append((source, sim))
        seen.add(source)

    twins.sort(key=lambda x: x[1], reverse=True)
    return twins


def _get_node_turn(graph, node_key: str) -> int:
    if node_key not in graph:
        return 0
    return max(0, int(graph.nodes[node_key].get("turn_id", 0)))


def _get_prior_turn_relations(
    graph,
    node_key: str,
    current_turn_count: int,
) -> list[tuple[str, str]]:
    rels = []
    if node_key not in graph:
        return rels
    for _, obj_key, data in graph.edges(node_key, data=True):
        if data.get("kind") != "fact" or data.get("quarantined", False):
            continue
        edge_turn = data.get("turn_id", 0)
        if edge_turn < current_turn_count:
            rels.append((data.get("rel", "").strip(), obj_key))

    for source, _, data in graph.in_edges(node_key, data=True):
        if data.get("kind") != "fact" or data.get("quarantined", False):
            continue
        edge_turn = data.get("turn_id", 0)
        if edge_turn < current_turn_count:
            rel = data.get("rel", "").strip()
            if rel:
                rels.append((rel, source))

    return [(r, o) for r, o in rels if r]


def _get_current_turn_relations(
    graph,
    node_key: str,
    current_turn_count: int,
) -> list[tuple[str, str]]:
    rels = []
    if node_key not in graph:
        return rels
    for _, obj_key, data in graph.edges(node_key, data=True):
        if data.get("kind") != "fact" or data.get("quarantined", False):
            continue
        if data.get("turn_id") == current_turn_count:
            rels.append((data.get("rel", "").strip(), obj_key))
    return [(r, o) for r, o in rels if r]


def _get_historical_relations(
    graph,
    node_key: str,
    current_turn_nodes: set,
) -> list[tuple[str, str]]:
    rels = []
    if node_key not in graph:
        return rels
    for _, obj_key, data in graph.edges(node_key, data=True):
        if data.get("kind") != "fact" or data.get("quarantined", False):
            continue
        if obj_key in current_turn_nodes:
            continue
        rel = data.get("rel", "").strip()
        if rel:
            rels.append((rel, obj_key))
    return rels


def _get_edge_turn_id(
    graph,
    sub_key: str,
    rel: str,
    obj_key: str,
) -> int:
    if sub_key not in graph or obj_key not in graph:
        return 0
    for _, neighbor, data in graph.edges(sub_key, data=True):
        if (neighbor == obj_key
                and data.get("kind") == "fact"
                and data.get("rel", "").strip() == rel.strip()):
            return int(data.get("turn_id", 0))
    return 0


def _get_turn_text(turn_history: list[dict], turn_id: int) -> str:
    for entry in turn_history:
        if entry.get("turn") == turn_id:
            p = entry.get("prompt", "").strip()
            r = entry.get("response", "").strip()
            return f"User: {p}\nAssistant: {r}"
    return ""
