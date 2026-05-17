
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer

from skg.triple_extractor import extract_triples
from skg.graph_manager import init_graph, add_triples_to_graph
from skg.semantic_linker import link_semantic_nodes


TURN_COLORS = [
    "#A8D8EA",
    "#B5EAD7",
    "#FFDAC1",
    "#FFB7B2",
    "#C7CEEA",
    "#E2F0CB",
    "#F3E0C2",
    "#D4A5A5",
    "#A0C4FF",
    "#CAFFBF",
]

TURN_BORDER_COLORS = [
    "#4A90D9",
    "#52B788",
    "#E8A87C",
    "#E07A72",
    "#8E99C9",
    "#A3C75A",
    "#C9A96E",
    "#B07070",
    "#6FA3E8",
    "#7DC97D",
]


def build_graph(input_path: str, embedding_model):
    with open(input_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)


    session = datasets[0]
    turns = session.get("turns", [])
    session_id = session.get("id", "session")

    graph = init_graph()
    turn_count = 0

    for turn in turns:
        turn_count += 1
        prompt = turn.get("prompt", "")
        response = turn.get("response", "")

        turn_text = f"User: {prompt}\nAssistant: {response}"
        existing_node_keys = list(graph.nodes)

        triples = extract_triples(
            turn_text,
            embedding_model=embedding_model,
            existing_nodes=existing_node_keys,
        )
        new_nodes = add_triples_to_graph(graph, triples, turn_count)
        link_semantic_nodes(graph, new_nodes, embedding_model)

        print(f"  [Turn {turn_count}] {len(triples)} triples, {len(new_nodes)} new nodes "
              f"(graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")

    return graph, turn_count, session_id


def render_graph(graph, turn_count: int, session_id: str, output_path: str):


    fact_edges = []
    semantic_edges = []

    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("kind") == "semantic":
            semantic_edges.append((u, v, key, data))
        else:
            fact_edges.append((u, v, key, data))

    print(f"\n  Graph: {graph.number_of_nodes()} nodes | "
          f"{len(fact_edges)} factual edges | {len(semantic_edges)} semantic edges")


    fig, ax = plt.subplots(1, 1, figsize=(20, 14), facecolor="white")
    ax.set_facecolor("white")
    ax.set_title(
        f"Semantic Knowledge Graph (SKG)\nSession: {session_id}",
        fontsize=16, fontweight="bold", color="#2C3E50", pad=20,
        fontfamily="serif",
    )


    pos = nx.spring_layout(
        graph, k=2.5, iterations=80, seed=42, scale=2.0,
    )


    node_colors = []
    node_edge_colors = []
    node_sizes = []
    labels = {}

    for node_key in graph.nodes:
        ndata = graph.nodes[node_key]
        turn_id = ndata.get("turn_id", 1)
        color_idx = min(turn_id - 1, len(TURN_COLORS) - 1)
        node_colors.append(TURN_COLORS[color_idx])
        node_edge_colors.append(TURN_BORDER_COLORS[color_idx])


        degree = graph.degree(node_key)
        node_sizes.append(max(800, 400 + degree * 120))


        label = ndata.get("label", node_key)

        if len(label) > 20:
            words = label.split()
            mid = len(words) // 2
            label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        labels[node_key] = label


    if semantic_edges:
        sem_edge_list = [(u, v) for u, v, _, _ in semantic_edges]
        nx.draw_networkx_edges(
            graph, pos, edgelist=sem_edge_list, ax=ax,
            edge_color="#B71C1C",
            style="solid",
            width=1.2,
            alpha=0.55,
            arrows=False,
            connectionstyle="arc3,rad=0.15",
        )


    if fact_edges:
        fact_edge_list = [(u, v) for u, v, _, _ in fact_edges]
        nx.draw_networkx_edges(
            graph, pos, edgelist=fact_edge_list, ax=ax,
            edge_color="#2980B9",
            style="dashed",
            width=1.8,
            alpha=0.7,
            arrows=True,
            arrowsize=15,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            min_source_margin=15,
            min_target_margin=15,
        )


    edge_labels = {}
    for u, v, key, data in fact_edges:
        pair = (u, v)
        rel = data.get("rel", "")
        if pair not in edge_labels and rel:

            if len(rel) > 25:
                rel = rel[:22] + "..."
            edge_labels[pair] = rel

    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, ax=ax,
        font_size=6,
        font_color="#2C3E50",
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
        rotate=True,
    )


    nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=2.0,
        alpha=0.92,
    )


    nx.draw_networkx_labels(
        graph, pos, labels=labels, ax=ax,
        font_size=7,
        font_weight="bold",
        font_color="#1A1A2E",
        font_family="serif",
    )


    turn_patches = []
    for i in range(min(turn_count, len(TURN_COLORS))):
        turn_patches.append(
            mpatches.Patch(
                facecolor=TURN_COLORS[i],
                edgecolor=TURN_BORDER_COLORS[i],
                linewidth=1.5,
                label=f"Turn {i+1}",
            )
        )


    edge_legend = [
        Line2D([0], [0], color="#2980B9", linewidth=2, linestyle="dashed",
               label="Factual Edge"),
        Line2D([0], [0], color="#B71C1C", linewidth=1.5, linestyle="solid",
               label="Semantic Edge"),
    ]

    legend1 = ax.legend(
        handles=turn_patches,
        loc="upper left",
        title="Node Origin (Turn)",
        title_fontsize=9,
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#BDC3C7",
        framealpha=0.95,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=edge_legend,
        loc="lower left",
        title="Edge Types",
        title_fontsize=9,
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="#BDC3C7",
        framealpha=0.95,
    )


    stats_text = (
        f"Nodes: {graph.number_of_nodes()}  |  "
        f"Factual Edges: {len(fact_edges)}  |  "
        f"Semantic Edges: {len(semantic_edges)}  |  "
        f"Turns: {turn_count}"
    )
    ax.text(
        0.5, -0.02, stats_text,
        transform=ax.transAxes,
        fontsize=9, color="#7F8C8D",
        ha="center", va="top",
        fontfamily="serif",
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  ✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SKG Knowledge Graph Visualization")
    parser.add_argument("-i", "--input", required=True, help="Input JSON dataset")
    parser.add_argument("-o", "--output", default="skg_visualization.png", help="Output PNG file")
    args = parser.parse_args()

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    print(f"\nBuilding SKG from: {args.input}")
    graph, turn_count, session_id = build_graph(args.input, embedding_model)

    print(f"\nRendering visualization...")
    render_graph(graph, turn_count, session_id, args.output)


if __name__ == "__main__":
    main()
