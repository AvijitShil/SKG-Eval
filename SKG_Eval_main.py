
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import re
import json
import csv
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


from skg.triple_extractor import extract_triples
from skg.graph_manager import init_graph, add_triples_to_graph, get_local_subgraph, mark_user_deprecated
from skg.semantic_linker import link_semantic_nodes
from skg.consistency_checker import compute_consistency

from skg.relational_contradiction_checker import check_relational_contradiction


DEFAULT_MODEL = "gemma4:31b"
DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"


WEIGHT_LOCAL = 0.40
WEIGHT_CONSISTENCY = 0.20
WEIGHT_LOGIC = 0.40
LOCAL_RELEVANCE_THRESHOLD = 0.3
DECAY_ALPHA = 0.7
CONTRADICTION_NOISE_THRESHOLD = 0.10


SHORT_RESPONSE_WORD_THRESHOLD = 12


SHORT_WEIGHT_LOCAL = 0.50
SHORT_WEIGHT_CONSISTENCY = 0.10
SHORT_WEIGHT_LOGIC = 0.40


GENERAL_WEIGHT_LOCAL = 0.50
GENERAL_WEIGHT_CONSISTENCY = 0.20
GENERAL_WEIGHT_LOGIC = 0.30


TREND_BASE_LAMBDA = 5.0
TREND_REFERENCE_TURNS = 20


RECENCY_GAMMA = 0.1


LOW_CONSISTENCY_GATE = 0.45
LOCAL_FLOOR_WHEN_CONSISTENCY_LOW = 0.20


JOINT_WEAK_LOCAL         = 0.50
JOINT_WEAK_CONSISTENCY   = 0.45
JOINT_PENALTY_MULTIPLIER = 0.75


RELEVANCE_FLOOR = 0.50


SHORT_PROMPT_WORD_THRESHOLD = 10


QA_WEIGHT_LOCAL       = 0.65
QA_WEIGHT_CONSISTENCY = 0.05
QA_WEIGHT_LOGIC       = 0.30


SESSION_ANCHOR_DISCOUNT = 0.85


GEOM_MIN_REL_SIM   = 0.15
GEOM_MIN_OBJ_SIM   = 0.10
GEOM_REL_DIVERGE   = 0.35
GEOM_OBJ_DIVERGE   = 0.20


NEGATION_MARKERS = frozenset({
    "not", "no", "never", "doesn't", "does not", "cannot",
    "isn't", "is not", "aren't", "are not", "won't", "will not",
    "doesn't", "does not", "didn't", "did not", "has no", "have no"
})


def get_ollama_llm(model: str) -> OllamaLLM:
    return OllamaLLM(
        model=model,
        temperature=0.2,
        base_url="http://127.0.0.1:11434",

        top_p=0.9,
        num_predict=256,
        stop=["User:", "\nUser"],
        seed=42
    )


def call_ollama(prompt: str, model: str, history: Optional[list] = None) -> str:
    llm = get_ollama_llm(model)


    if history:
        context_text = "\n".join([
            f"User: {turn['prompt']}\nAssistant: {turn['response']}"
            for turn in history if 'response' in turn
        ])
        full_prompt = f"Previous conversation:\n{context_text}\n\nCurrent question: {prompt}"
    else:
        full_prompt = prompt

    response = llm.invoke(full_prompt)
    return (response or "").strip()


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "being", "been", "as", "at", "by",
    "from", "that", "this", "it", "its", "into", "than", "also", "can",
    "will", "should", "may", "might", "must", "do", "does", "did",
    "have", "has", "had"
}


def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_words(s: str) -> list[str]:
    s = normalize_text(s)
    words = re.findall(r"[a-z0-9]+", s)
    return [w for w in words if w not in STOPWORDS and len(w) >= 3]


def classify_response_mode(response: str) -> str:
    stripped = response.strip()
    words = stripped.split()
    if len(words) < SHORT_RESPONSE_WORD_THRESHOLD:
        return "short"
    return "general"


def classify_prompt_mode(prompt: str) -> str:
    if not prompt or not prompt.strip():
        return "qa"
    word_count = len(prompt.strip().split())
    if word_count < SHORT_PROMPT_WORD_THRESHOLD:
        return "qa"
    return "narrative"


REVISION_VERBS = frozenset({
    "change", "update", "revert", "edit", "remove", "add", "replace",
    "swap", "switch", "modify", "revise", "redo", "undo", "fix",
    "correct", "adjust", "alter", "amend", "rework", "rename",
    "rewrite", "include", "exclude", "drop", "insert", "put",
    "make", "rebuild", "restructure", "reorganize", "reformulate",
    "overhaul", "refresh", "refine", "tweak", "trim", "cut",
    "shorten", "lengthen", "expand", "condense", "simplify",
    "clarify", "strengthen", "soften", "tighten", "loosen",
    "rephrase", "paraphrase", "reframe", "reorder", "reformat",
    "restate", "reword", "restore", "bring back", "go back",
    "go back to", "return to", "reset", "reinstate", "recover",
    "rather", "prefer", "instead", "without", "leave out",
    "do not include", "no longer", "should not", "should be",
    "want it", "want the", "would like", "i'd like", "i want",
    "i prefer", "i'd rather", "get rid of", "take out", "omit",
    "delete", "eliminate", "erase", "throw in", "append", "extend",
    "try again", "once more", "now make", "now change", "now add",
})

def is_revision_prompt(prompt: str, graph=None) -> bool:
    if not prompt: return False
    prompt_lower = prompt.lower().strip()

    first_60 = prompt_lower[:60]

    verb_found = False
    for verb in REVISION_VERBS:


        if re.search(rf'\b{re.escape(verb)}\b', first_60):
            verb_found = True
            break

    if not verb_found: return False


    if graph is not None and graph.number_of_nodes() > 0:
        return any(nk in prompt_lower for nk in graph.nodes if len(nk) > 3)
    return verb_found


def extract_revision_targets(
    prompt: str,
    triples: list[dict],
    graph,
    current_turn_id: int,
) -> set[str]:
    targets = set()
    prompt_lower = prompt.lower()

    PRONOUNS = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}


    first_words = set(w.strip(".,!?\"'") for w in prompt_lower.split()[:6])
    uses_pronoun = bool(first_words & PRONOUNS)

    if uses_pronoun:

        candidates = [
            (nk, data.get("importance", 0.0))
            for nk, data in graph.nodes(data=True)
            if data.get("turn_id", -1) < current_turn_id
            and nk not in PRONOUNS
            and len(nk) > 3
            and not data.get("quarantined", False)
        ]
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            targets.add(candidates[0][0])
        return targets


    for triple in triples:
        for key in ("sub", "obj"):
            entity = triple.get(key, "").lower().strip()
            if not entity or len(entity) <= 3:
                continue
            if entity not in prompt_lower:
                continue


            longest_match = max(
                (len(nk) for nk in graph.nodes if nk in prompt_lower),
                default=1,
            )
            if len(entity) / longest_match >= 0.60:
                targets.add(entity)

    return targets


class SessionEvaluator:

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        decay_alpha: float = DECAY_ALPHA,
        w_local: float = WEIGHT_LOCAL,
        w_consistency: float = WEIGHT_CONSISTENCY,
        w_logic: float = WEIGHT_LOGIC,
        llm_model: str = DEFAULT_MODEL,
    ):
        print(f"Loading embedding model: {embed_model}")
        self.embedding_model = SentenceTransformer(embed_model)


        self.context_vectors: list = []


        self.turn_history: list = []


        total_w = w_local + w_consistency + w_logic
        self.w_local = w_local / total_w
        self.w_consistency = w_consistency / total_w
        self.w_logic = w_logic / total_w


        self.turn_count = 0
        self.history_text = ""


        self.turn_mode_log: list = []


        self.graph = init_graph()
        self.llm_model = llm_model


        print("SKG initialized")

    def reset(self):
        self.context_vectors = []
        self.turn_history = []
        self.turn_count = 0
        self.history_text = ""
        self.turn_mode_log = []
        self.graph = init_graph()
        self.session_anchor_embedding = None

    def reset_context(self):
        self.context_vectors = []
        self.turn_history = []
        self.turn_count = 0
        self.history_text = ""
        self.turn_mode_log = []
        self.graph = init_graph()
        self.session_anchor_embedding = None
        print("--- Context Memory Reset for New Session (SKG cleared) ---")

    def _encode(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text).reshape(1, -1)

    def score_local_relevance(
        self,
        prompt_embedding: np.ndarray,
        response_embedding: np.ndarray
    ) -> float:
        sim = cosine_similarity(prompt_embedding, response_embedding)[0][0]
        return float(max(0.0, min(1.0, sim)))

    def _score_semantic_triangle(
        self,
        prompt: str,
        response: str,
        reference_text: str,
        eval_mode: str = "general",
        prompt_mode: str = "narrative",
    ) -> float:

        if eval_mode == "short":
            text_to_encode = f"Prompt: {prompt} Response: {response}"
        else:
            text_to_encode = response


        raw_sentences = re.split(r'(?<=[.!?]) +', text_to_encode)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            sentences = [text_to_encode]

        prompt_embedding    = self.embedding_model.encode(prompt).reshape(1, -1)
        sentence_embeddings = self.embedding_model.encode(sentences)
        if sentence_embeddings.ndim == 1:
            sentence_embeddings = sentence_embeddings.reshape(1, -1)

        prompt_scores  = cosine_similarity(prompt_embedding, sentence_embeddings)[0]
        s_prompt_match = float(np.max(prompt_scores))


        reference_exists   = bool(reference_text and reference_text.strip())
        prompt_substantive = len(prompt.strip().split()) >= 10

        use_triangle = reference_exists and prompt_substantive

        if use_triangle:

            ref_embedding = self.embedding_model.encode(reference_text).reshape(1, -1)
            ref_scores    = cosine_similarity(ref_embedding, sentence_embeddings)[0]
            s_ref_match   = float(np.max(ref_scores))
            s_local       = (0.4 * s_prompt_match) + (0.6 * s_ref_match)
            print(
                f"    [S_LOCAL] triangle: prompt={s_prompt_match:.3f} "
                f"ref={s_ref_match:.3f} -> {s_local:.3f}"
            )
        else:

            s_local = s_prompt_match
            reason  = "no_reference" if not reference_exists else "short_prompt"
            print(
                f"    [S_LOCAL] prompt-only ({reason}): "
                f"s_prompt_match={s_prompt_match:.3f} -> {s_local:.3f}"
            )

        return float(max(0.0, min(1.0, s_local)))

    def _score_consistency_maxpool(self, response_embedding: np.ndarray) -> float:
        if not self.context_vectors:
            return 1.0


        history_matrix = np.vstack(self.context_vectors)


        similarities = cosine_similarity(response_embedding, history_matrix)[0]


        s_consistency = float(np.max(similarities))

        return max(0.0, min(1.0, s_consistency))


    def score_consistency(self, response_embedding: np.ndarray) -> float:
        return self._score_consistency_maxpool(response_embedding)


    def score_turn(self, prompt, response, reference_text="", update_context=True):
        self.turn_count += 1


        mode = classify_response_mode(response)
        print(f"    [MODE CLASSIFIER]: '{mode.upper()}' ({len(response.split())} words)")


        prompt_mode = classify_prompt_mode(prompt)
        print(f"    [PROMPT MODE]: '{prompt_mode.upper()}' ({len(prompt.strip().split())} prompt words)")


        both_short = (mode == "short" and prompt_mode == "qa")
        if both_short:
            print(f"    [BOTH SHORT] Short prompt + short response - gates suppressed.")


        if mode == "short":
            eval_target_text = f"Prompt: {prompt} Response: {response}"
        else:
            eval_target_text = response


        full_response_embedding = self._encode(eval_target_text)


        if self.turn_count == 1:
            self.session_anchor_embedding = full_response_embedding.copy()
            print(f"    [SESSION ANCHOR] Set from turn 1 embedding.")


        s_local = self._score_semantic_triangle(
            prompt, response, reference_text,
            eval_mode=mode,
            prompt_mode=prompt_mode,
        )


        turn_text = f"User: {prompt}\nAssistant: {response}" if mode != "short" else eval_target_text


        existing_node_keys = list(self.graph.nodes)

        triples = extract_triples(
            turn_text,
            embedding_model=self.embedding_model,
            existing_nodes=existing_node_keys,
        )
        new_nodes = add_triples_to_graph(self.graph, triples, self.turn_count)
        link_semantic_nodes(self.graph, new_nodes, self.embedding_model)
        print(f"    [SKG] Extracted {len(triples)} triples, {len(new_nodes)} new nodes "
              f"(graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")


        graph_consistency = compute_consistency(self.graph, new_nodes)

        if self.session_anchor_embedding is not None and self.turn_count > 1:


            anchor_sim = float(cosine_similarity(
                full_response_embedding,
                self.session_anchor_embedding,
            )[0][0]) * SESSION_ANCHOR_DISCOUNT

            s_consistency = max(graph_consistency, anchor_sim)
            print(
                f"    [S_CONSISTENCY] graph={graph_consistency:.3f} "
                f"anchor_sim={anchor_sim:.3f} (raw={anchor_sim/SESSION_ANCHOR_DISCOUNT:.3f} x {SESSION_ANCHOR_DISCOUNT}) "
                f"-> max={s_consistency:.3f}"
            )
        else:
            s_consistency = graph_consistency
            print(f"    [S_CONSISTENCY] graph={graph_consistency:.3f} (no anchor yet)")


        is_revision = is_revision_prompt(prompt, graph=self.graph)

        if is_revision:
            revision_targets = extract_revision_targets(
                prompt, triples, self.graph, self.turn_count
            )
            if revision_targets:
                print(f"    [REVISION] Targets: {revision_targets} — marking user_deprecated for this turn only")
                mark_user_deprecated(self.graph, revision_targets, self.turn_count)
            else:
                print(f"    [REVISION] No targets found — full check runs")


        geom_rel_sim = 0.30 if prompt_mode == "qa" else 0.50
        s_logic = check_relational_contradiction(
            self.graph,
            new_nodes,
            triples,
            self.embedding_model,
            turn_history=self.turn_history,
            current_turn_text=turn_text,
            geom_min_rel_sim=geom_rel_sim,
        )


        if mode == "short":
            w_local       = SHORT_WEIGHT_LOCAL
            w_consistency = SHORT_WEIGHT_CONSISTENCY
            w_logic       = SHORT_WEIGHT_LOGIC
            profile_name  = "SHORT"
        elif prompt_mode == "qa":
            w_local       = QA_WEIGHT_LOCAL
            w_consistency = QA_WEIGHT_CONSISTENCY
            w_logic       = QA_WEIGHT_LOGIC
            profile_name  = "QA"
        else:
            w_local       = GENERAL_WEIGHT_LOCAL
            w_consistency = GENERAL_WEIGHT_CONSISTENCY
            w_logic       = GENERAL_WEIGHT_LOGIC
            profile_name  = "GENERAL"


        total         = w_local + w_consistency + w_logic
        w_local       /= total
        w_consistency /= total
        w_logic       /= total
        print(f"    [WEIGHTS] Profile={profile_name}: local={w_local:.3f} consistency={w_consistency:.3f} logic={w_logic:.3f}")


        quality_score = (w_local * s_local) + (w_consistency * s_consistency) + (w_logic * s_logic)


        if s_logic < 0.60:
            quality_score = min(quality_score, 0.40)
            print(
                f"   [!] S_LOGIC HARD GATE: s_logic ({s_logic:.3f}) < 0.60 "
                f"-> score capped at {quality_score:.3f}"
            )


        joint_weakness_fired = False
        if not both_short:
            if s_local < JOINT_WEAK_LOCAL and s_consistency < JOINT_WEAK_CONSISTENCY:
                quality_score *= JOINT_PENALTY_MULTIPLIER
                joint_weakness_fired = True
                print(
                    f"   [!] JOINT WEAKNESS GATE: s_local ({s_local:.3f}) < {JOINT_WEAK_LOCAL} "
                    f"AND s_consistency ({s_consistency:.3f}) < {JOINT_WEAK_CONSISTENCY} "
                    f"-> score penalised x{JOINT_PENALTY_MULTIPLIER} = {quality_score:.3f}"
                )


        is_non_sequitur = False

        if not both_short:
            if s_consistency < LOW_CONSISTENCY_GATE:
                if s_local < LOCAL_FLOOR_WHEN_CONSISTENCY_LOW:
                    quality_score *= 0.5
                    is_non_sequitur = True
                    print(
                        f"   [!] LOW CONSISTENCY GATE: s_consistency ({s_consistency:.3f}) < {LOW_CONSISTENCY_GATE} "
                        f"AND s_local ({s_local:.3f}) < {LOCAL_FLOOR_WHEN_CONSISTENCY_LOW} "
                        f"-> score softened x0.5 = {quality_score:.3f}"
                    )


        if update_context:

            if quality_score < 0.40:
                for node in new_nodes:
                    if node in self.graph.nodes:
                        self.graph.nodes[node]["quarantined"] = True
                for u, v, key, data in list(self.graph.edges(keys=True, data=True)):
                    if data.get("turn_id") == self.turn_count:
                        self.graph.edges[u, v, key]["quarantined"] = True
                print(f"   [!] QUARANTINE: Q(t)={quality_score:.3f} < 0.40 -> {len(new_nodes)} SKG nodes quarantined")


            if quality_score > 0.5 or s_logic > 0.8:
                self.context_vectors.append(full_response_embedding.flatten())


            self.turn_history.append({
                "turn": self.turn_count,
                "prompt": prompt,
                "response": response,
                "expected": reference_text,
                "s_logic": s_logic,
                "quality_score": quality_score
            })


            self.turn_mode_log.append(mode)


            self.history_text += f"\nUser: {prompt}\nAssistant: {response}"


        return {
            "turn":                 self.turn_count,
            "eval_mode":            mode,
            "prompt_mode":          prompt_mode,
            "both_short":           both_short,
            "is_revision":          is_revision,
            "s_local":              float(s_local),
            "s_consistency":        float(s_consistency),
            "s_logic":              float(s_logic),
            "quality_score":        float(quality_score),
            "is_non_sequitur":      is_non_sequitur,
            "joint_weakness_fired": joint_weakness_fired,
            "weights": {
                "w_local":       w_local,
                "w_consistency": w_consistency,
                "w_logic":       w_logic,
            },
            "response_vector": full_response_embedding,
        }


def load_conversation(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Conversation file not found: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))


    if isinstance(data, dict) and "turns" in data:
        return data["turns"]


    elif isinstance(data, list):

        if len(data) > 0 and isinstance(data[0], dict) and "turns" in data[0]:

            all_turns = []
            turn_counter = 0
            for conv in data:
                conv_id = conv.get("id", "unknown")
                print(f"Loading conversation: {conv_id}")
                for turn in conv.get("turns", []):
                    turn_counter += 1

                    turn_copy = dict(turn)
                    turn_copy["turn"] = turn_counter
                    turn_copy["conversation_id"] = conv_id
                    all_turns.append(turn_copy)
            return all_turns
        else:

            return data
    else:
        raise ValueError("Invalid conversation format")


def convert_csv_to_conversation(csv_path: str, output_path: str = None) -> list[dict]:
    df = pd.read_csv(csv_path)

    conversation = []
    history = []

    for idx, row in df.iterrows():
        turn = {
            "turn": int(idx) + 1,
            "prompt": str(row["prompt"]),
            "expected": str(row.get("expected", "")),

            "response": str(row["response"]) if "response" in row else None,
            "history": history.copy()
        }
        conversation.append(turn)


        history.append({
            "prompt": str(row["prompt"]),
            "response": str(row.get("expected", ""))
        })

    if output_path:
        Path(output_path).write_text(
            json.dumps(conversation, indent=2),
            encoding="utf-8"
        )
        print(f"Saved conversation to: {output_path}")

    return conversation


def plot_trajectory_graph(df, output_path: str = "trajectory_plot.png"):

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1]})


    ax1.axhspan(0.8, 1.05, color='green', alpha=0.15, label='Healthy Zone (>0.8)')
    ax1.axhspan(0.0, 0.5, color='red', alpha=0.15, label='Critical Zone (<0.5)')

    ax1.plot(df['global_turn'], df['quality_score'], marker='o', markersize=8,
             linewidth=2.5, color='blue', label='Actual Quality Q(t)')

    sessions = df['conversation_id'].unique()

    for i, session in enumerate(sessions):
        subset = df[df['conversation_id'] == session]

        if len(subset) > 1:
            n = len(subset)
            w = compute_recency_weights(n, gamma=RECENCY_GAMMA)
            z = np.polyfit(subset['global_turn'], subset['quality_score'], 1, w=w)
            p = np.poly1d(z)
            ax1.plot(subset['global_turn'], p(subset['global_turn']),
                     linestyle='--', linewidth=3, color='#D32F2F', alpha=0.9,
                     label='Recency-Weighted Trend' if i==0 else "")

        if i < len(sessions) - 1:
            sep_x = subset['global_turn'].max() + 0.5
            ax1.axvline(x=sep_x, color='gray', linestyle=':', linewidth=2)
            ax2.axvline(x=sep_x, color='gray', linestyle=':', linewidth=2)
            ax1.text(sep_x, 1.02, "Session Reset", ha='center', fontsize=10,
                     color='gray', backgroundcolor='white')

    ax1.set_ylabel("Quality Score", fontsize=12, fontweight='bold')
    ax1.set_title("Conversation Trajectory (Dynamic)", fontsize=16, fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.set_ylim(-0.05, 1.1)


    ax2.plot(df['global_turn'], df['s_local'], marker='s', label='Relevance (s_local)',
             color='green', alpha=0.8)
    ax2.plot(df['global_turn'], df['s_logic'], marker='d', label='Logic (s_logic)',
             color='#00BCD4', alpha=0.8)
    ax2.plot(df['global_turn'], df['s_consistency'], marker='^', label='Consistency (s_consistency)',
             color='purple', alpha=0.8)

    ax2.axhline(y=0.60, color='green', linestyle=':', linewidth=1.5, label='Relevance Gate (0.60)')
    ax2.axhline(y=0.45, color='purple', linestyle=':', linewidth=1.5, label='Consistency Gate (0.45)')

    ax2.set_ylabel("Component Scores", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Global Turn Number", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)

    ax2.legend(loc='lower left', ncol=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved dynamic trajectory plot: {output_path}")


def compute_recency_weights(num_turns: int, gamma: float = RECENCY_GAMMA) -> np.ndarray:
    positions = np.arange(num_turns)
    raw_weights = np.exp(gamma * positions)
    weights = raw_weights / raw_weights.sum()
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Unified Vector-Logic Trajectory Evaluator with Adaptive Lambda"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to conversation JSON or CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        default="trajectory_results.csv",
        help="Output CSV path for results"
    )
    parser.add_argument(
        "--plot",
        default="trajectory_plot.png",
        help="Output path for trajectory visualization"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--embed_model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBED_MODEL})"
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=DECAY_ALPHA,
        help=f"Context decay alpha (default: {DECAY_ALPHA})"
    )
    parser.add_argument(
        "--w_local",
        type=float,
        default=WEIGHT_LOCAL,
        help=f"Weight for local relevance (default: {WEIGHT_LOCAL})"
    )
    parser.add_argument(
        "--w_consistency",
        type=float,
        default=WEIGHT_CONSISTENCY,
        help=f"Weight for consistency (default: {WEIGHT_CONSISTENCY})"
    )
    parser.add_argument(
        "--w_logic",
        type=float,
        default=WEIGHT_LOGIC,
        help=f"Weight for logic (default: {WEIGHT_LOGIC})"
    )
    parser.add_argument(
        "--pass_threshold",
        type=float,
        default=0.6,
        help="Threshold for pass/fail (default: 0.6)"
    )
    parser.add_argument(
        "--trend_lambda",
        type=float,
        default=TREND_BASE_LAMBDA,
        help=f"Trend base lambda (default: {TREND_BASE_LAMBDA})"
    )
    parser.add_argument(
        "--convert_csv",
        action="store_true",
        help="Convert input CSV to conversation JSON format"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Offline mode: Use existing 'response' from dataset instead of generating with Ollama."
    )

    args = parser.parse_args()


    input_path = Path(args.input)

    if args.convert_csv or input_path.suffix.lower() == ".csv":
        print(f"Converting CSV to conversation format: {args.input}")
        json_path = input_path.with_suffix(".json")

        conversations = [{"id": "csv_import", "turns": convert_csv_to_conversation(args.input)}]
    else:

        data = json.loads(input_path.read_text(encoding="utf-8"))


        if isinstance(data, dict) and "turns" in data:
            conversations = [{"id": data.get("id", "single"), "turns": data["turns"]}]
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "turns" in data[0]:
                conversations = data
            else:
                conversations = [{"id": "legacy", "turns": data}]
        else:
            raise ValueError("Invalid conversation format")

    total_turns = sum(len(c.get("turns", [])) for c in conversations)
    print(f"\nLoaded {len(conversations)} conversation(s) with {total_turns} total turns")
    print(f"Adaptive Lambda Mode: lambda_base = {args.trend_lambda}, ref_turns = {TREND_REFERENCE_TURNS}")
    print("=" * 60)


    evaluator = SessionEvaluator(
        embed_model=args.embed_model,
        decay_alpha=args.decay,
        w_local=args.w_local,
        w_consistency=args.w_consistency,
        w_logic=args.w_logic,
        llm_model=args.model,
    )


    all_scores = []
    records = []
    session_stats = []
    global_turn_counter = 0

    for conversation in conversations:
        conversation_id = conversation.get("id", "unknown")
        print(f"\n{'=' * 60}")
        print(f"=== Processing Session: {conversation_id} ===")
        print(f"{'=' * 60}")


        evaluator.reset_context()


        history = []


        session_turns = []
        session_qualities = []
        session_start_idx = len(all_scores)

        for turn in conversation.get("turns", []):
            global_turn_counter += 1
            prompt = turn["prompt"]
            expected = turn.get("expected", "")
            local_turn = turn.get("turn", global_turn_counter)

            print(f"\n[Turn {global_turn_counter}] ({conversation_id}) Processing: {prompt[:50]}...")


            if args.offline:
                if "response" in turn and turn["response"]:
                    response = turn["response"]
                    print(f"    [Offline]: Using pre-generated response ({len(response)} chars)")
                else:
                    print("    [!] Warning: --offline flag set but no 'response' found in data. Skipping.")
                    response = ""
            else:

                response = call_ollama(prompt, args.model, history=history)


            scores = evaluator.score_turn(prompt, response, reference_text=expected)


            response_vector = scores.pop("response_vector", None)

            all_scores.append(scores)


            session_turns.append(global_turn_counter)
            session_qualities.append(scores["quality_score"])


            passed = 1 if scores["quality_score"] >= args.pass_threshold else 0


            if scores["s_local"] < RELEVANCE_FLOOR:
                passed = 0


            records.append({
                "global_turn":          global_turn_counter,
                "conversation_id":      conversation_id,
                "local_turn":           local_turn,
                "prompt":               prompt,
                "expected":             expected,
                "response":             response,
                "eval_mode":            scores["eval_mode"],
                "prompt_mode":          scores.get("prompt_mode", "narrative"),
                "both_short":           scores.get("both_short", False),
                "is_revision":          scores.get("is_revision", False),
                "s_local":              scores["s_local"],
                "s_consistency":        scores["s_consistency"],
                "s_logic":              scores["s_logic"],
                "quality_score":        scores["quality_score"],
                "is_non_sequitur":      scores["is_non_sequitur"],
                "joint_weakness_fired": scores.get("joint_weakness_fired", False),
                "passed":               passed,
                "model":                "",

                "session_slope":        "",
                "session_trend":        "",
                "session_turn_count":   "",
                "session_pass_count":   "",
                "session_pass_rate":    "",
                "recency_weighted_avg": "",
                "effective_lambda":     "",
            })


            print(f"    [{scores['eval_mode'].upper()}] S_local={scores['s_local']:.3f} | "
                  f"S_consistency={scores['s_consistency']:.3f} | "
                  f"S_logic={scores['s_logic']:.3f} -> "
                  f"Q(t)={scores['quality_score']:.3f} "
                  f"{'PASS' if passed else 'FAIL'}")

            if scores["is_non_sequitur"]:
                print(f"    [!] WARNING: Non-sequitur detected (low local relevance)")


            history.append({"prompt": prompt, "response": response})


        if evaluator.turn_history:
            hist_filename = f"history_dataset_{conversation_id}.csv"
            history_df = pd.DataFrame(evaluator.turn_history)
            history_df.to_csv(hist_filename, index=False)
            print(f"    [Saved History]: {hist_filename}")


        if len(session_qualities) >= 2:

            n = len(session_qualities)
            weights = compute_recency_weights(n, gamma=RECENCY_GAMMA)
            y = np.array(session_qualities)


            recency_weighted_avg = float(np.dot(weights, y))


            X = np.arange(n, dtype=float)
            coeffs = np.polyfit(X, y, deg=1, w=weights)
            slope = coeffs[0]
            intercept = coeffs[1]


            effective_lambda = args.trend_lambda * (n / TREND_REFERENCE_TURNS)


            final_score = recency_weighted_avg + (effective_lambda * slope)
            final_score = max(0.0, min(1.0, final_score))


            if slope > 0.01:
                trend_dir = "IMPROVING"
            elif slope < -0.01:
                trend_dir = "DEGRADING"
            else:
                trend_dir = "STABLE"


            session_stats.append({
                "session_id": conversation_id,
                "turns": session_turns,
                "slope": slope,
                "intercept": intercept,
                "final_score": final_score,
                "trend": trend_dir,
                "effective_lambda": effective_lambda,
                "recency_weighted_avg": recency_weighted_avg
            })


            print(f"\n    >> RECENCY-WEIGHTED PREDICTOR RESULTS ({conversation_id}):")
            print(f"       Slope (beta): {slope:+.4f}  |  Intercept (alpha): {intercept:.4f}")
            print(f"       Recency-Weighted Avg: {recency_weighted_avg:.4f}")
            print(f"       Effective Lambda (lambda): {effective_lambda:.2f}  |  Trend: {trend_dir}")
            print(f"       Final Score: {final_score:.4f}")


            session_turn_records = [
                r for r in records
                if r.get("conversation_id") == conversation_id
                and r.get("eval_mode") != "session"
            ]
            session_pass_count = sum(1 for r in session_turn_records if r.get("passed", 0) == 1)
            session_turn_count = len(session_turn_records)

            records.append({
                "global_turn":          "SESSION_SUMMARY",
                "conversation_id":      conversation_id,
                "local_turn":           "SUMMARY",
                "prompt":               "",
                "expected":             "",
                "response":             "",
                "eval_mode":            "session",
                "prompt_mode":          "",
                "both_short":           "",
                "is_revision":          "",
                "s_local":              round(float(np.mean([r["s_local"] for r in session_turn_records])), 4) if session_turn_records else 0.0,
                "s_consistency":        round(float(np.mean([r["s_consistency"] for r in session_turn_records])), 4) if session_turn_records else 0.0,
                "s_logic":              round(float(np.mean([r["s_logic"] for r in session_turn_records])), 4) if session_turn_records else 0.0,
                "quality_score":        round(float(final_score), 4),
                "is_non_sequitur":      "",
                "joint_weakness_fired": "",
                "passed":               1 if final_score >= args.pass_threshold else 0,
                "model":                "",
                "session_slope":        round(float(slope), 6),
                "session_trend":        trend_dir,
                "session_turn_count":   session_turn_count,
                "session_pass_count":   session_pass_count,
                "session_pass_rate":    round(session_pass_count / session_turn_count, 3) if session_turn_count > 0 else 0.0,
                "recency_weighted_avg": round(float(recency_weighted_avg), 4),
                "effective_lambda":     round(float(effective_lambda), 4),
            })
        else:

            single_q = session_qualities[0] if session_qualities else 0.5
            slope             = 0.0
            intercept         = single_q
            final_score       = single_q
            trend_dir         = "SINGLE TURN"
            recency_weighted_avg = single_q
            effective_lambda  = 0.0

            session_stats.append({
                "session_id":           conversation_id,
                "turns":                session_turns,
                "slope":                0.0,
                "intercept":            single_q,
                "final_score":          single_q,
                "trend":                "SINGLE TURN",
                "effective_lambda":     0.0,
                "recency_weighted_avg": single_q,
            })


            session_turn_records = [
                r for r in records
                if r.get("conversation_id") == conversation_id
                and r.get("eval_mode") != "session"
            ]
            session_pass_count = sum(1 for r in session_turn_records if r.get("passed", 0) == 1)

            records.append({
                "global_turn":          "SESSION_SUMMARY",
                "conversation_id":      conversation_id,
                "local_turn":           "SUMMARY",
                "prompt":               "",
                "expected":             "",
                "response":             "",
                "eval_mode":            "session",
                "prompt_mode":          "",
                "both_short":           "",
                "is_revision":          "",
                "s_local":              round(float(session_qualities[0]), 4) if session_qualities else 0.0,
                "s_consistency":        "",
                "s_logic":              "",
                "quality_score":        round(float(final_score), 4),
                "is_non_sequitur":      "",
                "joint_weakness_fired": "",
                "passed":               1 if final_score >= args.pass_threshold else 0,
                "model":                "",
                "session_slope":        0.0,
                "session_trend":        "SINGLE TURN",
                "session_turn_count":   1,
                "session_pass_count":   session_pass_count,
                "session_pass_rate":    float(session_pass_count),
                "recency_weighted_avg": round(float(single_q), 4),
                "effective_lambda":     0.0,
            })


    df = pd.DataFrame(records)


    summary_cols = [
        "session_slope", "session_trend", "session_turn_count",
        "session_pass_count", "session_pass_rate",
        "recency_weighted_avg", "effective_lambda",
    ]
    for col in summary_cols:
        if col not in df.columns:
            df[col] = ""


    global_avg_recency = np.mean([s["recency_weighted_avg"] for s in session_stats]) if session_stats else 0.0
    df["global_mean_recency_score"] = round(float(global_avg_recency), 4)

    df.to_csv(args.output, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    print(f"\n[OK] Saved results: {args.output}")
    print(f"  -> {len(df[df['eval_mode'] != 'session'])} turn rows")
    print(f"  -> {len(df[df['eval_mode'] == 'session'])} session summary rows")


    turn_df = df[df["eval_mode"] != "session"].copy()
    for col in ["global_turn", "quality_score", "s_local", "s_logic", "s_consistency"]:
        turn_df[col] = pd.to_numeric(turn_df[col], errors="coerce")
    plot_trajectory_graph(turn_df, args.plot)


    avg_quality = np.mean([s["quality_score"] for s in all_scores])

    turn_records = [r for r in records if r.get("eval_mode") != "session"]
    pass_rate = sum(1 for r in turn_records if r["passed"]) / len(turn_records) if turn_records else 0


    avg_slope = np.mean([s["slope"] for s in session_stats]) if session_stats else 0.0
    avg_final = np.mean([s["final_score"] for s in session_stats]) if session_stats else avg_quality


    short_turns = sum(1 for r in records if r["eval_mode"] == "short")
    general_turns = sum(1 for r in records if r["eval_mode"] == "general")

    print("\n" + "=" * 60)
    print("                 TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"  Total Conversations:  {len(conversations)}")
    print(f"  Total Turns:          {len(all_scores)}")
    print(f"  Mode Distribution:    {short_turns} SHORT / {general_turns} GENERAL")
    print("-" * 60)
    print(f"  Average Quality Q(t): {avg_quality:.4f}")
    print(f"  Pass Rate:            {pass_rate:.2%}")
    print(f"  Non-Sequiturs:        {sum(1 for s in all_scores if s['is_non_sequitur'])}")
    print("-" * 60)
    print("  >> RECENCY-WEIGHTED PREDICTOR ANALYSIS:")
    print(f"     Average Slope (beta): {avg_slope:+.4f}")
    print(f"     Final Score:       {avg_final:.4f}")
    print(f"     Base Lambda (lambda):   {args.trend_lambda}")


    if session_stats:
        lambdas_used = [s.get("effective_lambda", 0.0) for s in session_stats]
        if lambdas_used:
            print(f"     Lambda Range:      {min(lambdas_used):.2f} - {max(lambdas_used):.2f}")


    if session_stats:
        print("\n  Per-Session Breakdown:")
        for s in session_stats:
            print(f"    [{s['session_id']}] beta={s['slope']:+.4f}, "
                  f"S_final={s['final_score']:.4f}, lambda_eff={s.get('effective_lambda', 0.0):.2f}, "
                  f"{s['trend']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
