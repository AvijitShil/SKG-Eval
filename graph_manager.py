import networkx as nx
import numpy as np
from collections import deque

def init_graph() -> nx.MultiDiGraph:
    return nx.MultiDiGraph()

def add_triples_to_graph(
    graph: nx.MultiDiGraph,
    triples: list[dict],
    turn_id: int,
) -> list[str]:
    new_nodes: list[str] = []
    for triple in triples:
        sub_key = triple["sub"].lower().strip()
        obj_key = triple["obj"].lower().strip()
        triple_importance = float(triple.get("importance", 0.5))
        if not sub_key or not obj_key:
            continue
        if sub_key not in graph:
            graph.add_node(
                sub_key,
                label=triple["sub"],
                type=triple["type_sub"],
                embedding=None,
                importance=triple_importance,
                turn_id=turn_id,
                community_id=turn_id,
            )
            new_nodes.append(sub_key)
        else:
            if not graph.nodes[sub_key].get("type"):
                graph.nodes[sub_key]["type"] = triple["type_sub"]
            current_imp = graph.nodes[sub_key].get("importance", 0.0)
            graph.nodes[sub_key]["importance"] = max(current_imp, triple_importance)
        if obj_key not in graph:
            graph.add_node(
                obj_key,
                label=triple["obj"],
                type=triple["type_obj"],
                embedding=None,
                importance=triple_importance,
                turn_id=turn_id,
                community_id=turn_id,
            )
            new_nodes.append(obj_key)
        else:
            if not graph.nodes[obj_key].get("type"):
                graph.nodes[obj_key]["type"] = triple["type_obj"]
            current_imp = graph.nodes[obj_key].get("importance", 0.0)
            graph.nodes[obj_key]["importance"] = max(current_imp, triple_importance)
        graph.add_edge(
            sub_key,
            obj_key,
            rel=triple["rel"],
            kind="fact",
            turn_id=turn_id,
            rel_type=triple.get("rel_type", "assertion"),
            intent=triple.get("intent", "STATE"),
            attribute=triple.get("attribute", "property"),
            property_type=triple.get("property_type", "ADDITIVE"),
        )
    return new_nodes

def get_local_subgraph(
    graph: nx.MultiDiGraph,
    nodes: list[str],
    hops: int = 2,
) -> nx.MultiDiGraph:
    visited: set[str] = set()
    queue: deque = deque()
    for n in nodes:
        if n in graph:
            queue.append((n, 0))
            visited.add(n)
    while queue:
        current, depth = queue.popleft()
        if depth >= hops:
            continue
        neighbors = set(graph.successors(current)) | set(graph.predecessors(current))
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return graph.subgraph(visited).copy()

def mark_user_deprecated(
    graph: nx.MultiDiGraph,
    node_keys: set[str],
    current_turn_id: int,
) -> None:
    for node_key in node_keys:
        if node_key not in graph:
            continue
        for _, _, data in graph.edges(node_key, data=True):
            if data.get("kind") != "fact":
                continue
            if data.get("turn_id", -1) < current_turn_id:
                data["user_deprecated_until"] = current_turn_id

def get_edges_by_turn(
    graph: nx.MultiDiGraph,
    node_key: str,
    current_turn_id: int,
) -> tuple[list[dict], list[dict]]:
    curr_edges: list[dict] = []
    hist_edges: list[dict] = []
    if node_key not in graph:
        return curr_edges, hist_edges
    for _, obj_key, data in graph.edges(node_key, data=True):
        if data.get("kind") != "fact":
            continue
        if data.get("quarantined", False):
            continue
        deprecated_until = data.get("user_deprecated_until", -1)
        if deprecated_until >= current_turn_id:
            continue
        entry = {
            "rel":       data.get("rel", ""),
            "obj":       obj_key,
            "turn_id":   data.get("turn_id", -1),
            "rel_type":  data.get("rel_type", "assertion"),
            "intent":    data.get("intent", "STATE"),
            "attribute": data.get("attribute", "property"),
            "property_type": data.get("property_type", "ADDITIVE"),
        }
        if data.get("turn_id") == current_turn_id:
            curr_edges.append(entry)
        else:
            hist_edges.append(entry)
    return curr_edges, hist_edges
