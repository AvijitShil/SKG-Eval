
import os
import json
import re
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


_EXTRACTION_PROMPT = """\
You are an information extraction system for a multi-turn conversation evaluator.

Extract ONLY explicit factual subject-relation-object triples from the text.

EXTRACTION RULES:
- Do NOT infer anything not directly stated.
- Do NOT assume anything.
- Use ONLY what is explicitly said in the text.
- Keep relations short (1-4 words). Use verb phrases: "has", "causes", "is part of",
  "contradicts", "requires", "leads to".
- Assign a type to subject and object from:
  Person, Event, Object, Concept, Condition, Organization, Time, Number.
- Pay SPECIAL attention to comparisons and equivalencies. If the text says "A is the same as B" or "A is basically B", you MUST extract this as a triple: sub="A", rel="is the same as", obj="B".
- Pay SPECIAL attention to restrictive words like "only", "always", or "never". If the text says "calculated only on X", extract: sub="...", rel="is calculated only on", obj="X".
- ABSOLUTE SUBJECT RULE: NEVER use pronouns like "I", "me", "you", or "it" as a subject.
  If the speaker says "I bake bread," extract (bakery, bakes, bread).
  Always use the concrete noun representing the topic.
- EXCLUSIVE TAG RULE: If a fact describes a unique truth (Capitals, Atomic Numbers, Specific Counts, or the specific output of a business), you MUST use property_type: "EXCLUSIVE".

SUBJECT NORMALIZATION RULES (critical for contradiction detection across turns):
- Always use the most general, reusable form of the subject possible.
  CORRECT: "grief"         WRONG: "the grief process", "grief model", "stages of grief"
  CORRECT: "anxiety"       WRONG: "anxiety disorder", "chronic anxiety", "anxious state"
  CORRECT: "therapy"       WRONG: "cognitive behavioural therapy session"
  CORRECT: "memory"        WRONG: "traumatic memory", "the memory"
- If a concept was already named in an earlier part of the conversation,
  use that exact same name as the subject. Consistency across turns
  is more important than precision within a single turn.
- Never coin a new subject label when an existing general one fits.
  Ask yourself: "Is this concept already in the conversation under a simpler name?"
  If yes, use that simpler name.
- Avoid definite articles in subject/object labels.
  CORRECT: "five stages"   WRONG: "the five stages"
  CORRECT: "brain"         WRONG: "the brain"

SUBJECT STRIPPING RULES — apply these mechanically before writing the subject field:

  Strip leading action/gerund words — the ACTION is the relation, not part of the subject:
    "increasing model complexity" -> "model complexity"   (increasing = relation)
    "reducing overfitting"        -> "overfitting"        (reducing = relation)
    "adding regularization"       -> "regularization"     (adding = relation)
    "removing dropout"            -> "dropout"            (removing = relation)
    "applying pressure"           -> "pressure"           (applying = relation)
    "skipping breakfast"          -> "skipping breakfast" (EXCEPTION: the ACT of skipping IS the subject)

  Strip scalar/comparative qualifiers that modify the AMOUNT, not the nature:
    "larger learning rates"   -> "learning rate"   (larger = magnitude comparison, not the concept)
    "smaller batch size"      -> "batch size"
    "higher dropout rate"     -> "dropout rate"
    "lower model complexity"  -> "model complexity"
    "faster convergence"      -> "convergence"     (only if convergence is already a subject)
    RULE: If the qualifier is a comparative adjective (larger, smaller, higher, lower,
    faster, slower, greater, lesser, stronger, weaker), strip it. The relation should
    encode the direction (e.g. rel="increases" or rel="improves").
    EXCEPTION: Keep qualifiers that change the IDENTITY of the concept:
    "deep learning" ≠ "learning" — keep as-is.
    "transfer learning" ≠ "learning" — keep as-is.

  Strip role-of constructs — the ENTITY is the subject, not its role:
    "capital of Australia"    -> "Australia"     (put "is capital" in the relation)
    "president of France"     -> "France"        (put "has president" in the relation)
    "effect of learning rate" -> "learning rate" (put "causes" in the relation)
    "role of glucose"         -> "glucose"       (put "plays role in" in the relation)
    "impact of stress"        -> "stress"        (put "impacts" in the relation)
    "cause of inflation"      -> "inflation"     (put "is caused by" in the relation)

  Strip location qualifiers from physical/scientific subjects:
    "water at sea level"          -> "water"     (put condition in object or relation)
    "gas at high pressure"        -> "gas"
    "patient in ICU"              -> "patient"
    "student in advanced class"   -> "student"

  Strip possessives and article-noun wrapping:
    "the body's metabolism"  -> "metabolism"
    "a person's memory"      -> "memory"
    "the system's output"    -> "output"

EXCEPTION TO ALL STRIPPING: If the stripped form would be too generic to be
meaningful, keep the shortest meaningful noun phrase.

CRITICAL CROSS-TURN CONSISTENCY RULE:
- When a claim is about the EFFECT of something, the subject must be the CAUSE.
  CORRECT: "skipping breakfast" slows "metabolism"
  WRONG:   "body" slows "metabolism"  (when the context is about breakfast skipping)
- The subject should be the specific thing being discussed in this turn,
  not a generic biological entity.

RELATION NORMALIZATION RULES (CRITICAL FOR CONTRADICTION DETECTION):
You MUST map the relationship to one of the following strict, normalized verbs whenever possible. Do NOT invent new verbs if one of these fits the semantic meaning.

1. CAUSAL & DIRECTIONAL (For attributes: effect, property):
   - "increases" / "decreases"
   - "improves" / "worsens"
   - "causes" / "prevents"
   - "helps" / "harms"
   - "accelerates" / "slows"

2. DEFINITIONAL & COMPARATIVE (For attributes: definition, comparison):
   - "is" / "is not"
   - "has" / "does not have"
   - "requires" / "does not require"
   - "is the same as" / "is different from"
   - "equals" / "differs from"

3. FREQUENCY (If the text uses absolute terms like "always" or "never"):
   - "always causes" / "never causes"
   - "always improves" / "never improves"

FEW-SHOT EXAMPLES FOR NORMALIZATION:
Example 1 (Raw Text): "Yes, if the learning rate is too large, it can cause the optimization to diverge."
-> WRONG REL: "can cause"
-> CORRECT REL: "causes" (Map to strict causal list)

Example 2 (Raw Text): "Larger learning rates always improve convergence speed."
-> WRONG REL: "improve" (Misses the absolute frequency)
-> CORRECT REL: "always improves"

Example 3 (Raw Text): "Compound interest is basically the same as simple interest."
-> WRONG REL: "is basically"
-> CORRECT REL: "is the same as" (Map to strict comparative list)

ATTRIBUTE CLASSIFICATION (required -- assign exactly one per triple):

The "attribute" field names WHAT ASPECT of the subject is being described.
This is different from "rel" (which is HOW the claim is stated).
Two triples about the SAME subject with the SAME attribute are directly
comparable for contradiction. Two triples with DIFFERENT attributes describe
different aspects and are never contradictory.

ATTRIBUTE TAXONOMY -- use EXACTLY these strings, nothing else:

"definition"
    What the subject fundamentally IS. Its nature, meaning, or identity.
    Use when the response answers "what is X?" or "X is defined as..."
    EXAMPLES:
      sub="compound interest"  rel="is"  obj="interest on principal and prior interest"  -> "definition"
      sub="overfitting"        rel="is"  obj="memorizing training data"                 -> "definition"
      sub="Australia"          rel="has capital" obj="Canberra"                         -> "definition"
      sub="water"              rel="boils at"    obj="100 degrees Celsius"              -> "definition"
    KEY RULE: Use "definition" when the claim establishes identity, membership,
    or a fixed factual property that is not about what the subject DOES.

"effect"
    What the subject CAUSES, PRODUCES, or LEADS TO as an outcome.
    Use when the response describes consequences, results, or downstream impacts.
    EXAMPLES:
      sub="large learning rate" rel="causes"     obj="overshooting minimum"    -> "effect"
      sub="large learning rate" rel="leads to"   obj="faster convergence"      -> "effect"
      sub="overfitting"         rel="causes"     obj="poor generalization"      -> "effect"
      sub="skipping breakfast"  rel="slows"      obj="metabolism"              -> "effect"
    KEY RULE: The subject is the CAUSE. The object is the OUTCOME.

"property"
    A characteristic, feature, or trait describing HOW the subject behaves.
    EXAMPLES:
      sub="learning rate"     rel="controls"     obj="step size"               -> "property"
      sub="regularization"    rel="reduces"      obj="model complexity"        -> "property"
      sub="compound interest" rel="grows"        obj="exponentially"           -> "property"
    KEY RULE: "property" describes behavior or mechanism, not downstream outcome.
    DISAMBIGUATION from "effect": "effect" has a clear downstream OUTCOME.
      "learning rate controls step size" = property (mechanism)
      "large learning rate causes divergence" = effect (outcome for training)

"comparison"
    How the subject ranks, compares to, or differs from another entity.
    EXAMPLES:
      sub="compound interest" rel="is better than"   obj="simple interest"     -> "comparison"
      sub="compound interest" rel="is same as"       obj="simple interest"     -> "comparison"
    KEY RULE: The object is always another entity being compared TO.

"requirement"
    What the subject NEEDS, DEPENDS ON, or MUST HAVE to function.
    EXAMPLES:
      sub="gradient descent"  rel="requires"  obj="differentiable loss"        -> "requirement"

"quantity"
    A specific numerical value, measurement, or quantitative range.
    EXAMPLES:
      sub="water"         rel="boils at" obj="100 degrees Celsius at sea level" -> "quantity"
      sub="investment"    rel="returns"  obj="11000 after two years"            -> "quantity"
    KEY RULE: Use "quantity" when the object is a number, measurement, or rate.
    DISAMBIGUATION from "definition":
      If the object is a MEASUREMENT or NUMBER (temperature, weight, count, rate),
      use "quantity" EVEN IF the claim also defines the subject's nature.
      "water boils at 100°C" = quantity (100°C is a measurement)
      "water is a liquid" = definition (liquid is a category, not a number)

"negation"
    A claim that something does NOT apply or is FALSE.
    Only use when the claim explicitly negates a property, effect, or definition.
    EXAMPLES:
      sub="skipping breakfast" rel="does not affect"   obj="metabolism"        -> "negation"
      sub="compound interest"  rel="is not the same as" obj="simple interest"  -> "negation"
    KEY RULE: Use "negation" when "not", "no", "never", "does not" appears
    in the RELATION. Only for reversals of prior factual claims.

CRITICAL RULE -- ATTRIBUTE MUST MATCH ACROSS TURNS FOR COMPARISON:
Two triples about the same subject making COMPETING claims MUST have the same
attribute type. The detector ONLY compares triples with IDENTICAL attribute fields.
  WRONG (would miss contradiction):
    T1: sub="large LR" rel="leads to" obj="faster convergence" attribute="effect"
    T3: sub="large LR" rel="causes"   obj="divergence"         attribute="property"
  CORRECT (would catch contradiction):
    T1: sub="large LR" rel="leads to" obj="faster convergence" attribute="effect"
    T3: sub="large LR" rel="causes"   obj="divergence"         attribute="effect"

OBJECT RULES:
- Keep objects short and factual (1-5 words).
- Objects should be noun phrases, not full sentences.

OUTPUT QUANTITY RULES:
- Extract ALL subject-relation-object triples present in the text.
- There is NO limit on the number of triples. Extract as many as the text contains.
- A response with 5 sentences may produce 8-12 triples. That is correct.
- Do NOT stop early. Do NOT summarise multiple claims into one triple.
- Every distinct factual claim about a subject deserves its own triple.
- Even low-importance (0.1-0.4) claims must be extracted if they have
  a clear subject, relation, and object.

- For each triple, assign an importance score from 0.0 to 1.0:
    1.0 = this is the central claim of the response (the main fact being stated)
    0.7 = important supporting fact, directly relevant to the main topic
    0.4 = mentioned in passing, contextual detail
    0.1 = trivial or incidental mention (filler, example, aside)
  Be honest. Most triples in a long response are 0.4 or below.
  Only 1-2 triples per response should be 0.9 or above.

REL_TYPE CLASSIFICATION (required for every triple):
Assign one of these labels to "rel_type":

"assertion"          - A positive factual claim stated as true.
                       Example: sub="carbohydrates" rel="are" obj="preferred fuel" -> "assertion"

"negation_assertion" - A negated claim that REVERSES or CONTRADICTS a prior positive claim.
                       The subject is the SAME concept that was previously described differently.
                       Example: sub="skipping breakfast" rel="has no effect on" obj="metabolism"
                       when earlier the same concept was said to slow metabolism -> "negation_assertion"

"diagnosis"          - A description of a current problem, failure, or limitation.
                       This describes WHY something is failing, not what is true about it in general.
                       Example: sub="resume" rel="is not passing" obj="initial screening" -> "diagnosis"

"solution"           - A corrective action, recommendation, or technique to address a problem.
                       Example: sub="resume" rel="should mirror" obj="job keywords" -> "solution"

"elaboration"        - A supporting detail or expansion that adds information without conflicting.
                       Example: sub="protein" rel="has" obj="high satiety effect" -> "elaboration"

RULE: "negation_assertion" is ONLY correct when the negated claim directly reverses
a prior positive claim about the SAME concept in the SAME domain.
A problem description ("is not working", "is failing") is ALWAYS "diagnosis", never "negation_assertion".
A fix or recommendation is ALWAYS "solution", never "negation_assertion".

PROPERTY_TYPE CLASSIFICATION (required for every triple):
You MUST classify the logical nature of the relationship into one of two categories. Ask yourself: "Can this subject have multiple different values for this relation at the same time?"

Assign one of these labels to "property_type":

"EXCLUSIVE" - The subject can logically only have ONE of this object at a time. If a new object is introduced, it inherently overwrites the old one.
              Examples:
              - sub="France" rel="capital is" obj="Paris" -> EXCLUSIVE (a country has one capital)
              - sub="Oxygen" rel="has atomic number" obj="8" -> EXCLUSIVE (an element has one atomic number)
              - sub="The patient" rel="current heart rate is" obj="85 bpm" -> EXCLUSIVE (only one exact heart rate at a specific moment)

"ADDITIVE"  - The subject can have MULTIPLE of these objects simultaneously without conflict.
              CRITICAL RULE: Adjectives, subjective descriptions, opinions, and qualitative traits (e.g., "simple", "important", "beneficial") are ALWAYS ADDITIVE.
              Examples:
              - sub="Exercise" rel="is" obj="highly beneficial" -> ADDITIVE
              - sub="Stimulus control" rel="is" obj="simple" -> ADDITIVE
              - sub="Flu" rel="causes" obj="fever" -> ADDITIVE

Return ONLY valid JSON list in this format:

[
  {{
    "sub": "...",
    "rel": "...",
    "obj": "...",
    "type_sub": "...",
    "type_obj": "...",
    "importance": 0.0,
    "rel_type": "assertion",
    "attribute": "definition",
    "property_type": "EXCLUSIVE"
  }}
]

The "attribute" field must be exactly one of:
"definition", "effect", "property", "comparison", "requirement", "quantity", "negation"

Text:
{text}
"""


MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 7]

def extract_triples(
    turn_text: str,
    embedding_model=None,
    existing_nodes: list = None,
    *args,
    **kwargs,
) -> list[dict]:
    if not turn_text or not turn_text.strip():
        return []

    word_count = len(turn_text.split())
    client, model = _get_client()
    prompt = _EXTRACTION_PROMPT.format(text=turn_text)

    for attempt in range(MAX_RETRIES):
        raw = _call_llm(client, model, prompt, attempt)
        if raw is None:
            time.sleep(RETRY_DELAYS[attempt])
            continue

        result = _parse_triples(raw)


        if not result and word_count > 30:
            print(f"    [SKG] Attempt {attempt+1}: 0 triples from {word_count}-word response - retrying")
            time.sleep(RETRY_DELAYS[attempt])
            continue


        if result and embedding_model is not None:
            result = _deduplicate_subjects(result, embedding_model, existing_nodes)

        return result


    print(f"    [SKG] WARNING: All {MAX_RETRIES} attempts returned 0 triples. Running fallback.")
    result = _fallback_extraction(turn_text, client, model)
    if result and embedding_model is not None:
        result = _deduplicate_subjects(result, embedding_model, existing_nodes)
    return result


def _get_client():
    base_url = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key  = os.environ.get("LLM_API_KEY", "")
    model    = os.environ.get("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    return OpenAI(base_url=base_url, api_key=api_key), model


def _call_llm(client, model, prompt: str, attempt: int) -> str | None:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a structured data extraction system. Output only a valid JSON array. No explanation, no markdown, no preamble."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=1,
            max_tokens=4096,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"    [SKG] LLM call error (attempt {attempt+1}): {e}")
        return None


def _parse_triples(raw: str) -> list[dict]:
    if not raw:
        return []

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\[.*", raw, re.DOTALL)
    if not match:
        return []

    json_str = match.group()


    data = _try_parse_json(json_str)


    if data is None:
        json_str = _repair_truncated_json(json_str)
        data = _try_parse_json(json_str)

    if not isinstance(data, list):
        return []

    return _clean_triples(data)


def _try_parse_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _repair_truncated_json(s: str) -> str:
    last_close = s.rfind("}")
    if last_close == -1:
        return s
    repaired = s[:last_close + 1]

    opens  = repaired.count("[") - repaired.count("]")
    closes = repaired.count("{") - repaired.count("}")
    repaired += "}" * max(0, closes) + "]" * max(0, opens)
    return repaired


def _clean_triples(data: list) -> list[dict]:
    _VALID_REL_TYPES  = {"assertion", "negation_assertion", "diagnosis", "solution", "elaboration"}
    _VALID_INTENTS    = {"STATE", "ADVICE", "HYPOTHETICAL"}
    _VALID_ATTRIBUTES = {
        "definition", "effect", "property", "comparison",
        "requirement", "quantity", "negation"
    }
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue

        if not {"sub", "rel", "obj"}.issubset(item.keys()):
            continue
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        raw_rel_type = str(item.get("rel_type", "assertion")).strip().lower()
        rel_type = raw_rel_type if raw_rel_type in _VALID_REL_TYPES else "assertion"
        raw_intent = str(item.get("intent", "STATE")).strip().upper()
        intent = raw_intent if raw_intent in _VALID_INTENTS else "STATE"
        raw_attribute = str(item.get("attribute", "property")).strip().lower()
        attribute = raw_attribute if raw_attribute in _VALID_ATTRIBUTES else "property"
        raw_prop_type = str(item.get("property_type", "ADDITIVE")).strip().upper()
        property_type = raw_prop_type if raw_prop_type in {"EXCLUSIVE", "ADDITIVE"} else "ADDITIVE"
        cleaned.append({
            "sub":       str(item.get("sub", "")).strip(),
            "rel":       str(item.get("rel", "")).strip(),
            "obj":       str(item.get("obj", "")).strip(),
            "type_sub":  str(item.get("type_sub", "Concept")).strip(),
            "type_obj":  str(item.get("type_obj", "Concept")).strip(),
            "importance": max(0.0, min(1.0, importance)),
            "rel_type":  rel_type,
            "intent":    intent,
            "attribute": attribute,
            "property_type": property_type,
        })
    return cleaned


def _deduplicate_subjects(
    triples: list[dict],
    embedding_model,
    existing_nodes: list = None,
    similarity_threshold: float = 0.80,
) -> list[dict]:
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
    import numpy as np

    new_subjects = list({t["sub"] for t in triples if t.get("sub")})
    if not new_subjects:
        return triples

    new_embs = embedding_model.encode(new_subjects)
    if new_embs.ndim == 1:
        new_embs = new_embs.reshape(1, -1)
    canonical_map: dict[str, str] = {}


    if existing_nodes:
        safe_nodes = [n for n in existing_nodes if n and n.strip()]
        if safe_nodes:
            exist_embs = embedding_model.encode(safe_nodes)
            if exist_embs.ndim == 1:
                exist_embs = exist_embs.reshape(1, -1)
            cross_sim = _cos_sim(new_embs, exist_embs)
            for i, new_sub in enumerate(new_subjects):
                best_idx  = int(np.argmax(cross_sim[i]))
                best_sim  = float(cross_sim[i][best_idx])
                if best_sim >= similarity_threshold:
                    existing_key = safe_nodes[best_idx]
                    canonical_map[new_sub] = existing_key
                    print(
                        f"    [DEDUP] '{new_sub}' -> '{existing_key}' "
                        f"(cross-turn, sim={best_sim:.3f})"
                    )


    if existing_nodes:


        safe_nodes = sorted([n for n in existing_nodes if n and n.strip()], key=len, reverse=True)
        for new_sub in new_subjects:
            if new_sub in canonical_map:
                continue
            for exist_node in safe_nodes:

                if len(exist_node) < len(new_sub) and exist_node.lower() in new_sub.lower():
                    canonical_map[new_sub] = exist_node
                    print(f"    [DEDUP] '{new_sub}' -> '{exist_node}' (substring containment)")
                    break


    unmapped = [s for s in new_subjects if s not in canonical_map]
    if len(unmapped) > 1:
        unmapped_embs = embedding_model.encode(unmapped)
        if unmapped_embs.ndim == 1:
            unmapped_embs = unmapped_embs.reshape(1, -1)
        in_sim = _cos_sim(unmapped_embs, unmapped_embs)
        for i in range(len(unmapped)):
            for j in range(i + 1, len(unmapped)):
                if in_sim[i][j] >= similarity_threshold:
                    sub_i = unmapped[i]
                    sub_j = unmapped[j]
                    shorter = sub_i if len(sub_i) <= len(sub_j) else sub_j
                    longer  = sub_j if shorter == sub_i else sub_i
                    if longer not in canonical_map or len(shorter) < len(canonical_map[longer]):
                        canonical_map[longer] = shorter
                        print(
                            f"    [DEDUP] '{longer}' -> '{shorter}' "
                            f"(in-turn, sim={in_sim[i][j]:.3f})"
                        )

    if not canonical_map:
        return triples

    normalized = []
    for t in triples:
        new_t = dict(t)
        if new_t["sub"] in canonical_map:
            new_t["sub"] = canonical_map[new_t["sub"]]
        normalized.append(new_t)
    return normalized


_FALLBACK_PROMPT = """\
Extract subject-relation-object triples from the text below.
Return ONLY a JSON list. No explanation.

Format:
[{{"sub": "...", "rel": "...", "obj": "...", "type_sub": "Concept", "type_obj": "Concept", "importance": 0.5, "rel_type": "assertion", "attribute": "property"}}]

Text:
{text}
"""

def _fallback_extraction(turn_text: str, client, model: str) -> list[dict]:
    prompt = _FALLBACK_PROMPT.format(text=turn_text)
    raw = _call_llm(client, model, prompt, attempt=0)
    if raw is None:
        return []
    result = _parse_triples(raw)
    if result:
        print(f"    [SKG] Fallback extraction succeeded: {len(result)} triples")
    else:
        print(f"    [SKG] Fallback also returned 0 triples — turn will have no graph nodes")
    return result
