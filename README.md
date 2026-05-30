# SKG-Eval

Stateful evaluation of multi-turn dialogue using an incremental Semantic Knowledge Graph (SKG). Each conversational turn is parsed into subject-relation-object triples that are added to a growing graph. Three signals are computed per turn — local relevance, historical consistency, and logical coherence — and fused into a session-level quality score via recency-weighted trend analysis.

---

## Paper

**SKG-Eval: Stateful Evaluation of Multi-Turn Dialogue via Incremental Semantic Knowledge Graphs**

📄 arXiv: https://arxiv.org/abs/2605.16650

The paper introduces SKG-Eval, a quasi-deterministic and interpretable framework for evaluating multi-turn dialogue systems through an evolving Semantic Knowledge Graph (SKG). The framework incrementally tracks entities, relations, conversational commitments, contradictions, semantic drift, and historical consistency across dialogue turns.

---

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```



---

## Project Structure

```text
.
├── SKG_Eval_main.py
├── visualize_skg.py
├── probe.json
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

## Evaluation Datasets

Experiments were conducted using both publicly available benchmark datasets and a custom-designed diagnostic probe set.

### Public Benchmarks

#### MT-Bench-101

Repository: https://github.com/mtbench101/mt-bench-101

#### MultiChallenge

Repository: https://github.com/ekwinox117/multi-challenge

These benchmarks provide diverse multi-turn conversational tasks involving reasoning, instruction following, contextual understanding, dialogue management, and long-context interaction.

### SKG-Eval Probe Set

In addition to public benchmarks, we developed a dedicated probe set for evaluating conversational failure modes that are difficult to capture through conventional benchmark datasets.

The probe set contains manually curated multi-turn dialogue scenarios targeting:

- Direct contradictions
- Numeric inconsistencies
- Antonymic relation reversals
- Topic drift
- Semantic drift
- Historical inconsistency
- User-authorized revisions
- Long-range conversational memory failures

The probe set was specifically designed to stress-test the contradiction engine, historical consistency module, and state-tracking capabilities of SKG-Eval.

Example probe dialogues are provided in `probe.json`.

---

## Running the Evaluator

```bash
python SKG_Eval_main.py -i dataset.json -o results.csv --plot trajectory.png
```

### Arguments

| Argument | Description | Default |
|----------|-------------|----------|
| `-i` | Input JSON dataset | Required |
| `-o` | Output CSV file for per-turn and session scores | `results.csv` |
| `--plot` | Output PNG for the quality trajectory plot | `trajectory.png` |
| `--embed-model` | Sentence-Transformers model used for semantic embeddings | `all-mpnet-base-v2` |

Any compatible Sentence-Transformers embedding model may be used. Higher-quality embedding models may improve semantic linking and consistency estimation.

---

## Visualizing the Knowledge Graph

To render the SKG constructed from the first session in a dataset:

```bash
python visualize_skg.py -i dataset.json -o skg_graph.png
```

---

## Output

`results.csv` contains one row per turn plus a `SESSION_SUMMARY` row per session.

Key columns:

- `s_local` — local relevance score
- `s_consistency` — historical consistency score
- `s_logic` — logical coherence score
- `quality_score` — fused session score
- `session_slope` — trend direction (positive = improving)
- `session_trend` — `IMPROVING`, `STABLE`, or `DEGRADING`
- `passed` — 1 if `quality_score` ≥ pass threshold

Per-session conversation history is saved as:

```text
history_dataset_<id>.csv
```

---

## Reproducibility

SKG-Eval is designed as a quasi-deterministic evaluation framework. Given:

- Fixed benchmark inputs
- Fixed triple extraction outputs
- Fixed embedding model
- Fixed detector thresholds

the scoring pipeline produces deterministic and reproducible evaluation results.

All graph-construction procedures, contradiction detectors, scoring modules, and aggregation mechanisms are implemented explicitly and are available in the source code.

---

## Citation

If you use SKG-Eval in your research, please cite:

```bibtex
@article{shil2026skgeval,
  title={SKG-Eval: Stateful Evaluation of Multi-Turn Dialogue via Incremental Semantic Knowledge Graphs},
  author={Shil, Avijit and Samui, Suman},
  journal={arXiv preprint arXiv:2605.16650},
  year={2026}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
