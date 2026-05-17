
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def link_semantic_nodes(
    graph,
    new_nodes: list[str],
    embedding_model,
    threshold: float = 0.50,
) -> None:
    if not new_nodes:
        return

    all_node_keys = list(graph.nodes)
    if len(all_node_keys) < 2:
        return


    for nk in new_nodes:
        if graph.nodes[nk].get("embedding") is None:
            node_data = graph.nodes[nk]
            text = f"{node_data.get('label', nk)} {node_data.get('type', '')}".strip()
            emb = embedding_model.encode(text)
            graph.nodes[nk]["embedding"] = emb


    existing_keys = [k for k in all_node_keys if k not in set(new_nodes)]


    for ek in existing_keys:
        if graph.nodes[ek].get("embedding") is None:
            node_data = graph.nodes[ek]
            text = f"{node_data.get('label', ek)} {node_data.get('type', '')}".strip()
            emb = embedding_model.encode(text)
            graph.nodes[ek]["embedding"] = emb

    if not existing_keys:
        return


    new_embs = np.array([graph.nodes[nk]["embedding"] for nk in new_nodes])
    exist_embs = np.array([graph.nodes[ek]["embedding"] for ek in existing_keys])

    if new_embs.ndim == 1:
        new_embs = new_embs.reshape(1, -1)
    if exist_embs.ndim == 1:
        exist_embs = exist_embs.reshape(1, -1)

    sim_matrix = cos_sim(new_embs, exist_embs)

    for i, nk in enumerate(new_nodes):
        for j, ek in enumerate(existing_keys):
            similarity = float(sim_matrix[i][j])
            if similarity > threshold:
                graph.add_edge(
                    nk,
                    ek,
                    rel="semantic_related",
                    kind="semantic",
                    turn_id=-1,
                    weight=similarity,
                )
