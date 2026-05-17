_SCORE_FACT_EDGE = 1.0
_SCORE_SEMANTIC_EDGE = 0.65
_SCORE_NO_EDGE = 0.2

def compute_consistency(graph, new_nodes: list[str]) -> float:
    if not new_nodes:
        return 1.0
    existing_nodes = set(graph.nodes) - set(new_nodes)
    if not existing_nodes:
        return 1.0
    weighted_score_sum = 0.0
    importance_sum = 0.0
    plain_score_sum = 0.0
    for node_key in new_nodes:
        if node_key not in graph:
            anchor = _SCORE_NO_EDGE
            importance = 0.5
        else:
            anchor = _score_single_node(graph, node_key, existing_nodes)
            importance = float(graph.nodes[node_key].get("importance", 0.5))
            importance = max(0.0, min(1.0, importance))
        weighted_score_sum += importance * anchor
        importance_sum += importance
        plain_score_sum += anchor
    if importance_sum == 0.0:
        plain_mean = plain_score_sum / len(new_nodes)
        return float(max(0.0, min(1.0, plain_mean)))
    weighted_mean = weighted_score_sum / importance_sum
    return float(max(0.0, min(1.0, weighted_mean)))

def _score_single_node(graph, node_key: str, existing_nodes: set[str]) -> float:
    has_fact = False
    has_semantic = False
    for _, target, data in graph.edges(node_key, data=True):
        if target in existing_nodes:
            kind = data.get("kind", "")
            if kind == "fact":
                has_fact = True
                break
            elif kind == "semantic":
                has_semantic = True
    if not has_fact:
        for source, _, data in graph.in_edges(node_key, data=True):
            if source in existing_nodes:
                kind = data.get("kind", "")
                if kind == "fact":
                    has_fact = True
                    break
                elif kind == "semantic":
                    has_semantic = True
    if has_fact:
        return _SCORE_FACT_EDGE
    elif has_semantic:
        return _SCORE_SEMANTIC_EDGE
    else:
        return _SCORE_NO_EDGE
