# SKG-Eval

Stateful evaluation of multi-turn dialogue using an incremental Semantic Knowledge Graph (SKG). Each conversational turn is parsed into subject-relation-object triples that are added to a growing graph. Three signals are computed per turn — local relevance, historical consistency, and logical coherence — and fused into a session-level quality score via recency-weighted trend analysis.

---

## Requirements

- Python 3.10+
- An OpenAI-compatible API key for triple extraction (set in `.env`)

Install dependencies:

```
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

---

## Project Structure

```
.
├── SKG_Eval_main.py
├── visualize_skg.py
├── probe.json                  # example dataset
├── requirements.txt
└── skg/
    ├── triple_extractor.py
    ├── graph_manager.py
    ├── semantic_linker.py
    ├── consistency_checker.py
    ├── logic_checker.py
    └── relational_contradiction_checker.py
```

Place the `skg/` module files inside a folder named `skg/` with an `__init__.py`.

---

## Input Format

The input is a JSON file containing a list of sessions. Each session has an `id` and a list of `turns`. Each turn has a `prompt` and optionally a `response` (if left empty, the model generates one at runtime).

```json
[
  {
    "id": "session_001",
    "turns": [
      { "turn": 1, "prompt": "...", "response": "" },
      { "turn": 2, "prompt": "...", "response": "" }
    ]
  }
]
```

See `probe.json` for a complete example.

---

## Running the Evaluator

```
python SKG_Eval_main.py -i dataset.json -o results.csv --plot trajectory.png
```

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `-i` | Input JSON dataset | required |
| `-o` | Output CSV file for per-turn and session scores | `results.csv` |
| `--plot` | Output PNG for the quality trajectory plot | `trajectory.png` |
| `--embed-model` | Sentence-transformers model for embeddings | `all-mpnet-base-v2` | can use any other better embediing model 


---

## Visualizing the Knowledge Graph

To render the SKG constructed from the first session in a dataset:

```
python visualize_skg.py -i dataset.json -o skg_graph.png
```

---

## Output

`results.csv` contains one row per turn plus a `SESSION_SUMMARY` row per session. Key columns:

- `s_local` — local relevance score
- `s_consistency` — historical consistency score
- `s_logic` — logical coherence score
- `quality_score` — fused session score
- `session_slope` — trend direction (positive = improving)
- `session_trend` — `IMPROVING` / `STABLE` / `DEGRADING`
- `passed` — 1 if `quality_score` ≥ pass threshold

Per-session conversation history is saved as `history_dataset_<id>.csv`.

---
## License

This project is released under the MIT License. See `LICENSE` for details.
