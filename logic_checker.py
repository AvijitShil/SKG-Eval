
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def check_logic_conflict(
    graph,
    new_triples: list[dict],
    embedding_model=None,
) -> float:
    if not new_triples:
        return 1.0

    if graph.number_of_edges() == 0:
        return 1.0

    penalties: list[float] = []


    for triple in new_triples:
        sub_key = triple["sub"].lower().strip()
        obj_key = triple["obj"].lower().strip()
        new_rel = triple["rel"]

        if not sub_key or not obj_key:
            continue


        existing_rels = _get_existing_relations(graph, sub_key, obj_key)


        if not existing_rels and embedding_model is not None:
            fuzzy_pairs = _get_fuzzy_relations(
                graph, sub_key, obj_key, embedding_model
            )
            existing_rels = [pair[0] for pair in fuzzy_pairs]

        if not existing_rels:
            continue


        new_sentence = f"{triple['sub']} {new_rel} {triple['obj']}"

        for old_rel in existing_rels:
            old_sentence = f"{triple['sub']} {old_rel} {triple['obj']}"


            if old_rel.lower().strip() == new_rel.lower().strip():
                continue


    historical_sentences = _serialize_all_graph_triples(graph, new_triples)

    if historical_sentences:
        premise = " ".join(historical_sentences)

        for triple in new_triples:
            hypothesis = _serialize_triple(triple)
            if not hypothesis:
                continue

    if not penalties:
        return 1.0


    return float(max(0.0, min(1.0, min(penalties))))


def _serialize_triple(triple: dict) -> str:
    sub = triple.get("sub", "")
    rel = triple.get("rel", "")
    obj = triple.get("obj", "")
    if not sub or not rel or not obj:
        return ""
    return f"{sub} {rel} {obj}."


def _serialize_all_graph_triples(graph, exclude_triples: list[dict]) -> list[str]:

    exclude_keys = set()
    for t in exclude_triples:
        sub = t.get("sub", "").lower().strip()
        obj = t.get("obj", "").lower().strip()
        rel = t.get("rel", "").lower().strip()
        exclude_keys.add((sub, rel, obj))

    sentences = []
    for u, v, data in graph.edges(data=True):
        if data.get("kind") != "fact":
            continue
        if data.get("quarantined", False):
            continue
        rel = data.get("rel", "")

        if (u.lower().strip(), rel.lower().strip(), v.lower().strip()) in exclude_keys:
            continue

        sub_label = graph.nodes[u].get("label", u) if u in graph.nodes else u
        obj_label = graph.nodes[v].get("label", v) if v in graph.nodes else v
        sentence = f"{sub_label} {rel} {obj_label}."
        sentences.append(sentence)

    return sentences


def _get_existing_relations(graph, sub_key: str, obj_key: str) -> list[str]:
    rels: list[str] = []


    if graph.has_node(sub_key) and graph.has_node(obj_key):
        if graph.has_edge(sub_key, obj_key):
            for key in graph[sub_key][obj_key]:
                edge_data = graph[sub_key][obj_key][key]
                if edge_data.get("kind") == "fact" and not edge_data.get("quarantined", False):
                    rels.append(edge_data.get("rel", ""))


    if graph.has_node(obj_key) and graph.has_node(sub_key):
        if graph.has_edge(obj_key, sub_key):
            for key in graph[obj_key][sub_key]:
                edge_data = graph[obj_key][sub_key][key]
                if edge_data.get("kind") == "fact" and not edge_data.get("quarantined", False):
                    rels.append(edge_data.get("rel", ""))

    return rels


def _get_fuzzy_relations(
    graph,
    sub_key: str,
    obj_key: str,
    embedding_model,
    similarity_threshold: float = 0.80,
) -> list[tuple[str, str]]:
    results = []

    if sub_key not in graph:
        return results

    obj_emb = embedding_model.encode(obj_key).reshape(1, -1)

    for _, neighbor, data in graph.edges(sub_key, data=True):
        if data.get("kind") != "fact":
            continue
        if data.get("quarantined", False):
            continue
        if neighbor == obj_key:
            continue

        neighbor_emb_cached = graph.nodes.get(neighbor, {}).get("embedding")
        if neighbor_emb_cached is not None:
            neighbor_emb = np.array(neighbor_emb_cached).reshape(1, -1)
        else:
            neighbor_emb = embedding_model.encode(neighbor).reshape(1, -1)

        similarity = float(cos_sim(obj_emb, neighbor_emb)[0][0])
        if similarity >= similarity_threshold:
            rel = data.get("rel", "")
            results.append((rel, neighbor))

    return results



