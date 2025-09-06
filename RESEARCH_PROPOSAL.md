# TerseTalk: A Typed JSONL Protocol for Token-Efficient Multi-Agent LLMs (v0.5) **and** a Scoped Topology-Adaptive Extension (v1.0)

> **Final Version with All Updates** — Conditional Approval & Revised 8-Week MLSys Plan  
> **Status:** ✅ APPROVED **with conditions**. Focus on **protocol + microbenchmarks** first; topology is **bonus**.  
> _Last updated: 2025-09-04_

## 0) Executive Summary

### Problem gap

Most multi-agent LLM systems pass **free-form natural language** between agents—verbose, costly in tokens, and brittle to parse. Prompt compression (e.g., **LLMLingua/-2**) reduces tokens, but treats messages as generic text rather than an **on-wire, typed inter-agent protocol** ([LLMLingua](https://arxiv.org/abs/2310.05736), [LLMLingua-2](https://arxiv.org/abs/2403.12968)).

### Our v0.5 proposal (protocol-only)

We introduce **TerseTalk-JSONL**: a compact, newline-delimited JSON protocol (**JSON Lines/NDJSON**) with **single-letter tags**, **soft token caps**, and **overflow pointers** (`M#`) into a bounded memory store. We evaluate against **Free-form** and **Free-form + LLMLingua/-2** on **HotpotQA** (multi-hop QA) and **GSM8K** (multi-step math) with metrics covering:

- **Quality vs. tokens (Pareto)**
- **Semantic preservation (BERTScore; BLEURT optional)**
- **Overflow & memory behavior** (reset per task)
- **(De)serialization latency & bytes-on-wire**
- **Throughput & latency (TPS, TTFT, ITL)**

**Core Innovation:** We use validation-driven structured generation via **Instructor & Pydantic**, which guarantees protocol compliance with battle-tested tools. This makes our system **highly reproducible** and focuses the contribution on the **protocol's design**, which enables faster parsing and token efficiency. We deliberately choose battle-tested, reproducible tools (Instructor, Pydantic, Ollama) to ensure our contribution—the protocol design—can be readily adopted and extended by the community.
**Risk-mitigation (built-in)**: We ship a **hybrid safety net** that chooses, per message, the cheaper of **TerseTalk-JSONL** and **Free-form+LLMLingua**. We also let LLMLingua compress long tag payloads (pre-overflow), produce overflow summaries, and compress on dereference. This guarantees non-inferior compression while preserving the protocol’s parseability/streaming wins.

### Why now (and why JSONL)

- **JSON Lines/NDJSON**: standard, streaming-friendly "one JSON value per line"; easy to validate/diff/pipe; **O(1) message boundary detection**
- **Typed lines** drop verbose keys and enforce concise structure, enabling **10× faster tag extraction**
- **Soft caps + overflow pointers** avoid catastrophic truncation while retaining full content via `M#` dereference (like footnotes)
- **Validation-driven generation** ensures 100% protocol compliance with a reproducible and robust tech stack.
- **Strong baselines**: LLMLingua/-2 for fair comparison against state-of-the-art compression

References: [JSON Lines](https://jsonlines.org/), [NDJSON spec](https://github.com/ndjson/ndjson-spec), [HotpotQA](https://aclanthology.org/D18-1259/), [GSM8K](https://arxiv.org/abs/2110.14168), [BERTScore](https://arxiv.org/abs/1904.09675), [BLEURT](https://aclanthology.org/2020.acl-main.704.pdf).

---

## CA. Conditional Approval Requirements (MUST-Have)

### CA.1 Critical Path

**Weeks 1–2 — Protocol Foundation**

- **Structured Output (CORE)**: Define **Pydantic models** for all TerseTalk tags. Use **Instructor** to reliably generate compliant JSONL with an **Ollama-served model** (e.g., Mistral-7B).
- **Density proxy**: Use **overflow frequency** as initial metric: `density = 1.0 - (overflow_lines / total_lines)`
- **Mixed-format detection**: Implement guardrails for format breaks
- **Target:** >90% protocol compliance on 100-sample set
- **LLMLingua hooks (CORE):** wire pre-overflow, overflow-summary, and deref compression toggles; default off, enabled in calibration.
- **Hybrid gate (CORE):** implement per-turn gate with `--hybrid --token-budget <int>` flags and logging.

**Weeks 3–4 — Microbenchmarks (CORE)**

- **MB-1:** ≥10× faster tag extraction from JSONL vs free-form.
- **MB-2:** ≥5× faster streaming boundary detection.
- **MB-3:** SerDe & bytes-on-wire: JSONL vs free-form (and optional MsgPack/Protobuf). (Supersedes grammar-TTFT MB.)

**Weeks 5–6 — Full Evaluation**

- Scale to ≥500 examples per task
- Show 25–35% token reduction with <2% quality loss
- Complete failure analysis

**Week 7 — Topology (Only if Ahead)**

- Implement overflow-frequency-triggered binary switch
- Demonstrate on HotpotQA & GSM8K only

**Week 8 — Paper Writing**

### CA.2 Non-Negotiable Requirements

1. **One baseline local model via Ollama (e.g., mistral:instruct)** One baseline model served via Ollama (Mistral-7B recommended for speed and quality).
2. **Mixed-format detection** with deterministic fallback
3. **Density calibration** on 50 samples
4. **Reproducibility infrastructure** with global seed control
5. **Memory reset per task** (no state leakage)
6. **Results organization** with versioning and cleanup

### CA.3 Risk Mitigations

- **Grammar fails:** Pivot to post-hoc correction, report overhead as contribution
- **Topology shows no benefit:** Frame as "protocol enables future optimization"
- **Time shortage:** Submit v0.5 only as 4-page short paper

---

## 1) What's new vs. prior art (gap analysis)

- **Typed protocol for inter-agent communication:** Most frameworks (AutoGen, CAMEL) pass free-form text. TerseTalk defines a **compact, typed on-wire protocol** designed for **minimal tokens** and **fast parseability**.  
  References: [AutoGen](https://arxiv.org/abs/2308.08155), [CAMEL](https://arxiv.org/abs/2303.17760).

- **Head-to-head vs. learned compression:** LLMLingua/-2 is message-agnostic; we test **typed protocol vs. learned compression** directly in inter-agent settings.  
  References: [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/), [LLMLingua-2](https://arxiv.org/abs/2403.12968).

- **Practical Structured Generation:** Recent benchmarks show constraints can **speed up** generation (Guidance: 50% faster). We're first to apply this to multi-agent protocols.

- **Systems-first evaluation:** Token accounting, task quality, semantic preservation, **and** systems metrics (TPS, TTFT, ITL, serialization overhead).

**Takeaway:** v0.5 is a **protocol-only systems contribution** with measurable wins; v1.0 adds minimal topology adaptation.

---

## 2) TerseTalk-JSONL Protocol Specification

### 2.1 Core Design Principles

- **One JSON value per line** (JSONL/NDJSON standard)
- **Canonical form**: array `[tag, payload…]`
- **Lenient parsing**: accept objects, normalize to arrays
- **Overflow mechanism**: store long content in memory, send pointer `M#`

### 2.2 Protocol Tags (single letters)

```python
TAGS = {
    "r": "role",         # ["r","M"|"W"|"C"] (M=Manager, W=Worker, C=Critic)
    "g": "subgoal",      # ["g","<short text>"] (soft cap ~30 tokens)
    "f": "fact",         # ["f","<text>","M#id"?] (cap ~30; optional M#)
    "u": "assumption",   # ["u","<short text>"] (cap ~20)
    "p": "plan_step",    # ["p","<short text>"] (cap ~20)
    "q": "question",     # ["q","M|W|C","<question>"] (cap ~30)
    "d": "delta_ref",    # ["d","M#id"] (reference to memory)
    "v": "verdict",      # ["v","A"|"R"|"E"] (Accept/Revise/Escalate)
    "o": "overflow",     # ["o","<summary>","M#ref","method"?] (cap ~40)
    "t": "free_text",    # ["t","<very short>"] (escape hatch, cap ~50)
    "x": "context"       # ["x","<key>","<val>"] (extensible metadata)
}
```

### 2.3 Example Messages

**Manager → Worker:**

```jsonl
["r","M"]
["g","Compare dates of two events; return earlier."]
["f","Event A: 2001-07-16"]
["f","Event B: 1999-05-02"]
["u","Use ISO dates"]
["p","Check both dates"]
["p","Return earlier only"]
["q","W","Which is earlier?"]
["d","M#12"]
```

**Overflow handling:**

```jsonl
["f","Mars Climate Orbiter..."]
["o","Mars orbiter summary","M#23","extractive"]
```

### 2.4 Memory References (`M#`)

- **Format:** `M#<id>` where id is a unique integer
- **Storage:** Bounded key-value store (10,000 entries max)
- **Eviction:** Oldest-first when at capacity
- **Reset:** Memory cleared between tasks (no state leakage)
- **Dereferencing:** Optional based on policy (always/conditional/never)

### 2.5 Normalization Rules

**Lenient → Canonical:**

```jsonl
{"f": "Event A: 2001"}                      → ["f","Event A: 2001"]
{"tag":"f","text":"Event A","id":"M#12"}    → ["f","Event A","M#12"]
```

---

## 3) System Architecture (v0.5)

### 3.1 Core Components

- **Topology:** Manager-coordinated pipeline (M→W→C flow, not true star)
- **Protocol:** TerseTalk-JSONL with soft caps and overflow
- **Memory:** Bounded store with `M#` references (reset per task)
- **Structured Generation:** Instructor + Pydantic for guaranteed compliance.
- **Baselines:** Free-form, Free-form + LLMLingua-2

### 3.2 Metrics Framework

**Task Performance:**

- HotpotQA: Exact Match (EM)
- GSM8K: Exact answer match
- Token reduction: % saved vs. free-form
- Quality preservation: maintained accuracy

**Systems Metrics:**

- **TPS:** Tokens per second throughput
- **TTFT:** Time to first token (target: 20-30% reduction)
- **ITL:** Inter-token latency
- **Serialization:** JSONL parse/emit overhead (microseconds)
- **Memory:** Bytes on wire, overflow rates

**Semantic Preservation:**

- **Primary:** BERTScore (if torch available)
- **Fallback:** Jaccard similarity
- **Optional:** BLEURT

### 3.3 Hybrid Safety Net (LLMLingua-in-the-loop)

**Three touchpoints (all toggleable):**

1. **Pre-overflow compression:** run LLMLingua on long ["f"|"p"|"q"] payloads to avoid emitting ["o",…] when possible.

2. **Overflow summarization:** when ["o", summary, "M#…"] is required, generate summary via extractive (default) or LLMLingua.

3. **Dereference compression:** when a downstream step requests full text (dereference_policy in {"always","conditional"}), compress M# contents before re-injection.

**Per-turn hybrid gate (cheap, deterministic):**

Estimate tokens for the outgoing JSONL vs. a free-form prompt; probe LLMLingua’s projected length once if over budget; route the current turn to the cheaper path.

**Logs:** decision, budgets, projected tokens; used in analysis & ablations.

Mermaid (gate logic):

```mermaid
flowchart LR
A[Manager JSONL] -->|estimate tokens| B{<= budget?}
B -- Yes --> T[TerseTalk JSONL → next agent]
B -- No --> L[LLMLingua compress<br/>(payloads or full prompt)]
L --> C{fits budget?}
C -- Yes --> F[Free-form+LLMLingua → next agent]
C -- No --> O[TerseTalk + overflow (o,M#) → next agent]
```

---

## 4) Detailed Implementation Plan

### 4.1 Week 1-2: Protocol Foundation

#### PR-00 — Reproducibility Infrastructure (~150 LOC)

**Context:** Ensure deterministic, reproducible experiments across all runs.

**Public API (`tersetalk/reproducibility.py`):**

```python
def set_global_seed(seed: int) -> dict:
    """Set seeds for all libraries, return model config"""
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['LLAMA_CPP_SEED'] = str(seed)

    return {
        'temperature': 0.0,  # Deterministic
        'seed': seed,
        'top_p': 1.0,
        'top_k': 1
    }
```

**DoD (Definition of Done):** All experiments produce identical results with same seed.

#### PR-01 — Repository Scaffold (~80 LOC)

**Before→After:** Empty → working skeleton
**Files:** `requirements.txt`, `README.md`, `Makefile`, `.gitignore`
**DoD:** `make install` works; `python scripts/run_v05.py --help` prints

#### PR-02 — JSONL Protocol & Validator (~220 LOC)

**Context:** Core protocol implementation with validation and overflow.

**Public API (`tersetalk/protocol_jsonl.py`):**

```python
class JSONLValidator:
    def __init__(self, caps: dict[str,int], memory: MemoryStore):
        self.caps = caps
        self.memory = memory
        self.overflow_freq = {}  # Track per-tag overflow rates

    def validate_and_overflow(self, jsonl: str) -> tuple[str, dict]:
        """Validate JSONL, apply caps, create overflow lines"""

    def detect_format_break(self, output: str) -> tuple[bool, int]:
        """Returns (is_mixed, line_number_of_break)"""
        lines = [ln for ln in output.splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            s = line.lstrip()
            if not (s.startswith('[') or s.startswith('{')):
                return True, i
        return False, -1

    def normalize_line(self, raw: str) -> list:
        """Convert lenient object to canonical array"""

    def jsonl_to_prose(self, lines: str) -> str:
        """Convert JSONL to natural language for SP reference"""

    def create_overflow(self, text: str, tag: str) -> tuple[str, str]:
        """Create summary and overflow line"""
        summary = self.summarize(text, tag)
        mem_id = self.memory.put(text)
        return summary, f'["o","{summary}","{mem_id}","extractive"]'
```

**DoD:** Mixed format detected; caps enforced; overflow logged.

#### PR-02S — Structured Output with Instructor (CORE) (~150 LOC)

**Context:** Use Pydantic and Instructor for robust, reproducible, and easy-to-maintain structured output. This replaces complex GBNF grammars.

**Public API (`tersetalk/structured.py`):**

```python
from pydantic import BaseModel, Field
from typing import Literal, Union, List

# Define a Pydantic model for a single TerseTalk line
class TerseTalkLine(BaseModel):
    tag: Literal["r", "g", "f", "u", "p", "q", "d", "v", "o", "t", "x"]
    payload: List[str]

def to_jsonl_array(self) -> List[Union[str, List[str]]]:
    return [self.tag] + self.payload

# Example of a more specific model for validation
class FactLine(BaseModel):
    tag: Literal["f"] = "f"
    text: str = Field(..., description="The factual statement.")
    ref: Optional[str] = Field(None, pattern=r"^M#\d+$")
```

**DoD:** >90% compliance rate; TTFT measurement shows improvement.

#### PR-03 — Memory Store (~140 LOC)

**Public API (`tersetalk/memory.py`):**

```python
class MemoryStore:
    MAX_ENTRIES = 10_000

    def __init__(self):
        self.store = {}
        self.access_times = {}
        self.counter = 0

    def put(self, text: str) -> str:
        """Store text, return M# reference"""
        if len(self.store) >= self.MAX_ENTRIES:
            self._evict_oldest()
        self.counter += 1
        mid = f"M#{self.counter}"
        self.store[mid] = text
        self.access_times[mid] = time.time()
        return mid

    def get(self, mid: str) -> str | None:
        """Retrieve text by M# reference"""
        if mid in self.store:
            self.access_times[mid] = time.time()
            return self.store[mid]
        return None

    def reset(self):
        """Clear all entries (between tasks)"""
        self.store.clear()
        self.access_times.clear()
        self.counter = 0

    def stats(self) -> dict:
        """Return memory usage statistics"""
        return {
            "entries": len(self.store),
            "bytes": sum(len(v) for v in self.store.values()),
            "oldest": min(self.access_times.values()) if self.access_times else None
        }
```

**DoD:** Bounded size; eviction works; reset clears state.

#### PR-04 — Summarization Module (~180 LOC)

**Context:** Create summaries for overflow lines.

**Public API (`tersetalk/summarization.py`):**

```python
class Summarizer:
    def __init__(self, method="extractive"):
        self.method = method
        if method == "llmlingua":
            from llmlingua import PromptCompressor
            self.compressor = PromptCompressor()

    def summarize(self, text: str, tag: str, target_tokens: int = 20) -> str:
        """Create summary for overflow"""
        if self.method == "extractive":
            return self._extractive_summary(text, target_tokens)
        elif self.method == "llmlingua":
            return self._llmlingua_summary(text, target_tokens)
        else:
            return text[:target_tokens * 4] + "..."

    def _extractive_summary(self, text: str, target_tokens: int) -> str:
        """TextRank-based extraction"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 2:
            return text[:target_tokens * 4]

        # Score sentences by TF-IDF, position, keywords
        scores = self._score_sentences(sentences)
        selected = self._select_top_sentences(sentences, scores, target_tokens)
        return ' '.join(selected)
```

**DoD:** Summaries fit within target tokens; preserve key information.

#### PR-H1 (CORE) — hybrid_gate.py (~120 LOC)

**API:** gate_choose_protocol(jsonl_str, free_prompt, cfg) -> {"route": "tersetalk"|"freeform_llmlingua", "est_tokens": {...}}

**Public API (`tersetalk/hybrid_gate.py`):**

```python
from dataclasses import dataclass

@dataclass
class GateCfg:
    token_budget: int = 600
    use_ll2_tags: tuple[str, ...] = ("f","p","q")

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def project_ll2_tokens(prompt: str, budget: int) -> int | None:
    try:
        from llmlingua import PromptCompressor
        comp = PromptCompressor()
        proj = comp.compress(prompt, target_token=budget)
        return proj.get("compressed_tokens", None) or proj.get("target_token", None)
    except Exception:
        return None

def gate_choose_protocol(manager_jsonl: str, freeform_prompt: str, cfg: GateCfg) -> dict:
    t_jsonl = estimate_tokens(manager_jsonl)
    if t_jsonl <= cfg.token_budget:
        return {"route": "tersetalk", "est_tokens": {"jsonl": t_jsonl}}

    t_proj = project_ll2_tokens(freeform_prompt, cfg.token_budget)
    if t_proj is not None and t_proj <= cfg.token_budget:
        return {"route": "freeform_llmlingua", "est_tokens": {"ll2": t_proj, "jsonl": t_jsonl}}

    return {"route": "tersetalk", "est_tokens": {"jsonl": t_jsonl, "ll2_proj": t_proj}}
```

**DoD:** unit test stub routes based on thresholds; logs decision.

#### PR-H2 (CORE) — calibrate_caps.py (~180 LOC)

Sweep {caps, summarizer, deref_policy, gate_on/off, token_budget} on a 50-example shard; write best combo to configs/calibration.yaml.

**DoD:** deterministic output; schema validated in tests.

#### PR-H3 (CORE) — analysis/noninferiority.py (~120 LOC)

Bootstrap CI for one-sided non-inferiority (δ=0.02).

**DoD:** returns decision + CI bounds; tested on synthetic accuracies.

#### PR-H4 (CORE) — protocol_handler.py update (~160 LOC)

Plug LLMLingua at pre-overflow, overflow summary, deref; expose flags in runner.

**DoD:** toggles respected; counters recorded.

### 4.2 Week 3-4: Microbenchmarks (CORE)

#### PR-MB — Microbenchmark Suite (~600 LOC total)

**Context:** MLSys requires "10× somewhere" claims.

**Structure (`benchmarks/`):**

```python
# benchmarks/tag_extraction.py
def benchmark_tag_extraction():
    """Compare JSONL vs free-form tag extraction"""
    jsonl_msgs = ['["f","fact1"]'] * 10000
    freeform_msgs = ['The fact is: fact1'] * 10000

    # JSONL: Direct array indexing
    start = time.perf_counter()
    for msg in jsonl_msgs:
        tag = json.loads(msg)[0]
    jsonl_time = time.perf_counter() - start

    # Free-form: Regex extraction
    start = time.perf_counter()
    for msg in freeform_msgs:
        match = re.search(r'(fact|goal|plan):', msg)
        tag = match.group(1) if match else None
    freeform_time = time.perf_counter() - start

    return {
        "jsonl_time": jsonl_time,
        "freeform_time": freeform_time,
        "speedup": freeform_time / jsonl_time
    }

# benchmarks/streaming_boundaries.py
def benchmark_streaming():
    """Message boundary detection in streams"""
    # JSONL: newline detection O(1)
    # Free-form: sentence boundary O(n)
```

**DoD:**

- Tag extraction: ≥10× speedup
- Streaming: ≥5× speedup
- Grammar: 20-30% TTFT reduction (or documented failure analysis)

### 4.3 Week 5-6: Full System Implementation

#### PR-05 — Model I/O with Instructor & Ollama (~120 LOC)

**Context**: A simple, reliable client for generating structured data using Instructor with an Ollama backend.

**Public API (`tersetalk/model_io.py`):**

```python
import instructor
from openai import OpenAI
from tersetalk.structured import TerseTalkLine # From PR-02S
from typing import List

# Patch the OpenAI client to use Instructor
# This works with any OpenAI-compatible server, including Ollama
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama", # required but unused
    )
)

class ModelClient:

    def init(self, model_name: str = "mistral"):
        self.model = model_name

    def call_jsonl_strict(self, system: str, user_prompt: str, max_tokens=256) -> List[TerseTalkLine]:
    """Generate a list of valid TerseTalk lines."""
        return client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            response_model=List[TerseTalkLine], # Magic!
            max_retries=2, # Built-in robustness
        )

class EchoModel(ModelClient):
    """For CI testing without a model server"""
    def call_jsonl_strict(self, system: str, user_prompt: str, max_tokens=256) -> List[TerseTalkLine]:
        # Return a fixed, valid response for testing
        return [TerseTalkLine(tag="g", payload=["This is an echoed goal."])]

class ManagerMessage(BaseModel):
    """Complete Manager message validation"""
    role: Literal["r"] = "r"
    goal: str = Field(..., max_length=120)  # ~30 tokens
    facts: List[str] = Field(..., max_items=10)
    question: str = Field(..., max_length=120)

    def to_jsonl(self) -> str:
        """Convert to TerseTalk JSONL format"""
        lines = [["r", "M"]]
        lines.append(["g", self.goal])
        for fact in self.facts:
            lines.append(["f", fact])
        lines.append(["q", "W", self.question])
        return "\n".join(json.dumps(line) for line in lines)
```

**DoD:** Strict JSONL enforced; Echo enables GPU-free CI.

#### PR-06 — Dataset Adapters (~200 LOC)

**Public API (`tersetalk/datasets.py`):**

```python
def load_hotpotqa(split="validation", n=100, seed=0) -> list[dict]:
    """Load HotpotQA examples"""
    dataset = load_dataset("hotpot_qa", split=split)

    # Deterministic sampling
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    examples = []
    for idx in indices:
        item = dataset[idx]
        examples.append({
            "question": item["question"],
            "answer": item["answer"],
            "facts": item["supporting_facts"],
            "subgoal": f"Answer: {item['question'][:50]}...",
            "assumptions": ["Use provided facts", "Be concise"]
        })

    return examples

def load_gsm8k(split="test", n=100, seed=0) -> list[dict]:
    """Load GSM8K examples"""
    # Similar structure
```

**DoD:** Deterministic loading; normalized format; fast execution.

#### PR-07 — Manager-Coordinated Pipeline (~230 LOC)

**Context:** Implement M→W→C flow (not true star topology).

**Public API (`tersetalk/pipeline_runner.py`):**

```python
def run_pipeline_once(example: dict, client: ModelClient, config: dict) -> dict:
    """
    Run Manager→Worker→Critic pipeline once.
    Note: This is a coordinated pipeline, not true star topology.
    """
    memory = MemoryStore()
    validator = JSONLValidator(config['caps'], memory)

    # Build Manager message
    manager_jsonl = build_manager_message(example)

    # SP reference (before any compression)
    sp_reference = validator.jsonl_to_prose(manager_jsonl)

    # Manager → Worker
    start = time.perf_counter()
    validated_jsonl, overflow_stats = validator.validate_and_overflow(manager_jsonl)
    worker_response = client.call_jsonl_strict("You are a Worker", validated_jsonl)
    worker_time = time.perf_counter() - start

    # Worker → Critic (through Manager coordination)
    critic_input = prepare_critic_input(worker_response)
    critic_response = client.call_jsonl_strict("You are a Critic", critic_input)

    # Extract verdict
    verdict = extract_verdict(critic_response)

    # Compute metrics
    tokens_total = validator.estimate_tokens(manager_jsonl) + \
                   validator.estimate_tokens(worker_response) + \
                   validator.estimate_tokens(critic_response)

    # Memory cleanup
    memory.reset()

    return {
        "answer": extract_answer(worker_response),
        "verdict": verdict,
        "tokens_total": tokens_total,
        "overflow_count": overflow_stats['count'],
        "latency_ms": {
            "manager": 0,
            "worker": worker_time * 1000,
            "critic": critic_time * 1000
        },
        "sp_reference": sp_reference,
        "memory_stats": memory.stats()
    }
```

**DoD:** Pipeline completes; memory resets; metrics logged.

#### PR-08 — Baselines (~230 LOC)

**Public API (`tersetalk/baselines.py`):**

```python
def run_freeform_once(example: dict, client: ModelClient) -> dict:
    """Free-form baseline"""
    prompt = f"""
    Role: Manager
    Goal: {example['subgoal']}
    Facts: {'; '.join(example['facts'])}
    Question: {example['question']}
    """
    response = client.call(prompt)
    return {"answer": response, "tokens": len(prompt)/4 + len(response)/4}

def run_llmlingua_once(example: dict, client: ModelClient) -> dict:
    """Free-form + LLMLingua compression"""
    from llmlingua import PromptCompressor
    compressor = PromptCompressor()

    prompt = build_freeform_prompt(example)
    compressed = compressor.compress(prompt, target_token=100)
    response = client.call(compressed['compressed_prompt'])

    return {
        "answer": response,
        "tokens": compressed['origin_tokens'],
        "compression_ratio": compressed['ratio']
    }
```

**DoD:** Both baselines produce compatible metrics.

#### PR-09 — Results Manager (~180 LOC) [NEW]

**Context:** Organized, versioned results storage.

**Public API (`tersetalk/results_manager.py`):**

```python
class ResultsManager:
    def __init__(self, base_dir="results"):
        self.base_dir = Path(base_dir)

    def get_run_dir(self, experiment_id: str, timestamp: bool = True):
        """Create versioned run directory"""
        if timestamp:
            run_id = f"{datetime.now():%Y-%m-%d-%H-%M-%S}"
        else:
            run_id = "latest"

        path = self.base_dir / experiment_id / run_id
        path.mkdir(parents=True, exist_ok=True)

        # Symlink latest
        latest_link = self.base_dir / experiment_id / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_id)

        return path

    def save_config(self, run_dir: Path, config: dict):
        """Save experiment configuration"""
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2)
        )

    def cleanup_old_runs(self, experiment_id: str, keep_last_n: int = 5):
        """Remove old runs, keeping recent and best"""
        exp_dir = self.base_dir / experiment_id
        runs = sorted(exp_dir.glob("20*"))  # Date-prefixed dirs

        if len(runs) > keep_last_n:
            for run in runs[:-keep_last_n]:
                shutil.rmtree(run)
```

**Directory structure:**

```
results/
├── hotpotqa/
│   ├── 2025-09-04-12-30-45/
│   │   ├── config.json
│   │   ├── raw_outputs.jsonl
│   │   ├── metrics.csv
│   │   └── summary.json
│   └── latest → 2025-09-04-12-30-45/
└── figures/
    ├── pareto_curves.pdf
    └── ablation_caps.pdf
```

**DoD:** Versioned dirs; config saved; cleanup works.

### 4.4 Week 5-6: Evaluation Framework

#### PR-10 — Metrics Module (~240 LOC)

**Public API (`tersetalk/metrics.py`):**

```python
class MetricsComputer:
    def __init__(self, use_tiktoken=False):
        self.use_tiktoken = use_tiktoken
        if use_tiktoken:
            import tiktoken
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens (exact or approximate)"""
        if self.use_tiktoken:
            return len(self.encoder.encode(text))
        return len(text) // 4  # Rule of thumb

    def exact_match(self, pred: str, gold: str) -> bool:
        """HotpotQA EM scoring"""
        return normalize_answer(pred) == normalize_answer(gold)

    def gsm8k_correct(self, pred: str, gold: str) -> bool:
        """GSM8K exact answer match"""
        pred_num = extract_number(pred)
        gold_num = extract_number(gold)
        return pred_num == gold_num

    def bertscore_sp(self, reference: str, candidate: str) -> float:
        """Semantic preservation via BERTScore"""
        try:
            from bert_score import score
            P, R, F1 = score([candidate], [reference], lang="en")
            return F1.item()
        except ImportError:
            return self.jaccard_sp(reference, candidate)

    def jaccard_sp(self, reference: str, candidate: str) -> float:
        """Fallback semantic preservation"""
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        if not ref_tokens:
            return 0.0
        return len(ref_tokens & cand_tokens) / len(ref_tokens | cand_tokens)

    def measure_generation_timing(self, start: float, ttft: float,
                                  end: float, tokens: int) -> dict:
        """Compute TPS, TTFT, ITL"""
        total_time = end - start
        ttft_ms = (ttft - start) * 1000

        if tokens > 1:
            itl_ms = ((end - ttft) / (tokens - 1)) * 1000
        else:
            itl_ms = 0

        tps = tokens / total_time if total_time > 0 else 0

        return {
            "tps": tps,
            "ttft_ms": ttft_ms,
            "itl_ms": itl_ms,
            "total_time_s": total_time
        }
```

CI red-lines (smoke set, n=50):

Compliance ≥ 95% (Instructor validation success).

Token saving ≥ 20% with gate on (if off <20%).

Non-inferiority passes for Hybrid (δ = 0.02).
Failures trigger an automatic re-run with gate on and stricter caps; if still failing, CI blocks merge.

**DoD:** All metrics compute correctly; fallbacks work.

#### PR-11 — Experiment Driver (~250 LOC)

**Public API (`scripts/run_v05.py`):**

```python
@click.command()
@click.option('--task', type=click.Choice(['hotpotqa', 'gsm8k']))
@click.option('--system', type=click.Choice(['tersetalk', 'freeform', 'llmlingua']))
@click.option('--n', default=100, help='Number of examples')
@click.option('--seed', default=0, help='Random seed')
@click.option('--caps', default='{"f":30,"p":20,"q":30}', help='Soft caps (JSON)')
@click.option('--model', default='mistral')
@click.option('--out', default='results', help='Output directory')
def main(task, system, n, seed, caps, grammar, model, out):
    """Run TerseTalk experiments"""

    # Set reproducibility
    config = set_global_seed(seed)
    config.update({
        'task': task,
        'system': system,
        'n': n,
        'seed': seed,
        'caps': json.loads(caps),
        'model': model
    })

    # Setup results
    results_mgr = ResultsManager(out)
    run_dir = results_mgr.get_run_dir(f"{task}_{system}")
    results_mgr.save_config(run_dir, config)

    # Load data
    if task == 'hotpotqa':
        examples = load_hotpotqa(n=n, seed=seed)
    else:
        examples = load_gsm8k(n=n, seed=seed)

    # Initialize model
    client = ModelClient(model) if model != 'echo' else EchoModel()

    # Run experiments
    results = []
    for i, example in enumerate(tqdm(examples)):
        if system == 'tersetalk':
            result = run_pipeline_once(example, client, config)
        elif system == 'freeform':
            result = run_freeform_once(example, client)
        else:
            result = run_llmlingua_once(example, client)

        results.append(result)

        # Save incrementally
        if i % 10 == 0:
            save_results(results, run_dir)

    # Final save and summary
    save_results(results, run_dir)
    print_summary(results)
```

**DoD:** Full pipeline runs; results saved; reproducible with seed.

#### PR-12 — Analysis Scripts (~220 LOC)

**Public API (`scripts/analyze_v05.py`):**

```python
def generate_pareto_curve(results_dir: Path):
    """Quality vs Tokens Pareto frontier"""
    systems = ['tersetalk', 'freeform', 'llmlingua']

    for system in systems:
        metrics = load_metrics(results_dir / system / 'latest')
        plt.scatter(metrics['tokens'], metrics['quality'], label=system)

    plt.xlabel('Total Tokens')
    plt.ylabel('Task Accuracy')
    plt.legend()
    plt.savefig(results_dir / 'figures' / 'pareto.pdf')

def generate_ablation_plots(results_dir: Path):
    """Ablation studies on caps, etc."""
    caps_configs = [
        {"f": 20, "p": 15, "q": 20},  # Aggressive
        {"f": 30, "p": 20, "q": 30},  # Baseline
        {"f": 50, "p": 40, "q": 50},  # Relaxed
        {"f": 100, "p": 80, "q": 100}, # Very relaxed
    ]

    results = []
    for caps in caps_configs:
        metrics = load_metrics_for_caps(results_dir, caps)
        results.append({
            'avg_cap': np.mean(list(caps.values())),
            'tokens': metrics['tokens'],
            'quality': metrics['quality'],
            'overflow_rate': metrics['overflow_rate']
        })

    # Plot trade-offs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([r['avg_cap'] for r in results],
             [r['tokens'] for r in results], 'o-')
    ax1.set_xlabel('Average Cap Size')
    ax1.set_ylabel('Total Tokens')

    ax2.plot([r['overflow_rate'] for r in results],
             [r['quality'] for r in results], 'o-')
    ax2.set_xlabel('Overflow Rate')
    ax2.set_ylabel('Task Quality')

    plt.savefig(results_dir / 'figures' / 'ablation_caps.pdf')
```

**DoD:** Figures generated; insights clear; publication-ready.

#### PR-13 — Baselines Parameters & Robustness (~200 LOC)

```python
# tersetalk/baselines.py (update existing)

def run_freeform_once(example: dict, client: ModelClient, max_tokens: int = 256) -> dict:
"""
Free-form baseline with configurable max_tokens.

    Args:
        example: Task example with question, facts, etc.
        client: Model client for generation
        max_tokens: Maximum tokens for response (default 256)
    """
    prompt = build_freeform_prompt(example)

    try:
        response = client.call(prompt, max_tokens=max_tokens)
        tokens_used = estimate_tokens(prompt) + estimate_tokens(response)
    except Exception as e:
        # Record failure but don't crash
        return {
            "answer": "",
            "tokens": estimate_tokens(prompt),
            "error": str(e),
            "status": "error"
        }

    return {
        "answer": response,
        "tokens": tokens_used,
        "status": "success"
    }

def run_llmlingua_once(example: dict, client: ModelClient,
max_tokens: int = 256,
target_compression: int = 100) -> dict:
"""
LLMLingua baseline with configurable parameters.

    Args:
        max_tokens: Max tokens for model response
        target_compression: Target tokens after compression
    """
    prompt = build_freeform_prompt(example)

    try:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor()
        compressed = compressor.compress(
            prompt,
            target_token=target_compression
        )

        response = client.call(
            compressed['compressed_prompt'],
            max_tokens=max_tokens
        )

        return {
            "answer": response,
            "tokens": compressed['origin_tokens'],
            "compressed_tokens": compressed['compressed_tokens'],
            "compression_ratio": compressed['ratio'],
            "status": "success"
        }
    except ImportError:
        # LLMLingua not available, fall back to free-form
        return run_freeform_once(example, client, max_tokens)
    except Exception as e:
        return {
            "answer": "",
            "tokens": estimate_tokens(prompt),
            "error": str(e),
            "status": "error"
        }

def build_freeform_prompt(example: dict) -> str:
"""
Build free-form prompt from example.

    Note: The standalone "Question:" line is retained to match
    typical multi-agent prompting patterns where the query
    is clearly delineated from context.
    """
    facts_str = '\n'.join(f"- {fact}" for fact in example.get('facts', []))

    prompt = f"""Role: Manager

Goal: {example.get('subgoal', 'Answer the question')}

Facts:
{facts_str}

Assumptions:
{'; '.join(example.get('assumptions', []))}

Question: {example['question']}"""

    return prompt
```

##### scripts/baselines_smoke.py (new file)

```python
import click
from tersetalk.baselines import run_freeform_once, run_llmlingua_once
from tersetalk.model_io import ModelClient, EchoModel

@click.command()
@click.option('--model', type=click.Choice(['echo', 'real']), default='echo')
@click.option('--max-tokens', default=256)
@click.option('--target-compression', default=100)
def main(model, max_tokens, target_compression):
    """Smoke test for baseline systems."""

    client = EchoModel() if model == 'echo' else ModelClient()

    example = {
        "question": "What is 2+2?",
        "facts": ["Basic arithmetic", "Addition operation"],
        "subgoal": "Solve the math problem",
        "assumptions": ["Use standard arithmetic"]
    }

    # Test free-form
    result_ff = run_freeform_once(example, client, max_tokens)
    print(f"Free-form: {result_ff}")

    # Test LLMLingua
    result_ll = run_llmlingua_once(
        example, client, max_tokens, target_compression
    )
    print(f"LLMLingua: {result_ll}")

if __name__ == "__main__":
    main()
```

**Tests (tests/test_baselines_params.py):**

```python
pythondef test_max_tokens_parameter():
    """Verify max_tokens is passed through correctly."""
    client = EchoModel()
    example = {"question": "test", "facts": []}

    result = run_freeform_once(example, client, max_tokens=128)
    assert result["status"] == "success"

def test_error_handling():
    """Verify baselines don't crash on errors."""
    # Test with a client that raises exceptions
    # Verify we get error status but no crash
```

**DoD:** Baselines accept parameters; errors handled gracefully; smoke CLI works.

#### PR-14+EVAL — Real Experiments & Evaluation Driver (~250 LOC)

Combining experiment setup with evaluation driver since they're tightly coupled

```python
# scripts/run_evaluation.py (new file)

import click
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random

from tersetalk.datasets import load_hotpotqa, load_gsm8k
from tersetalk.pipeline_runner import run_pipeline_once
from tersetalk.baselines import run_freeform_once, run_llmlingua_once
from tersetalk.model_io import ModelClient, EchoModel
from tersetalk.reproducibility import set_global_seed

@click.command()
@click.option('--task', type=click.Choice(['hotpotqa', 'gsm8k']), required=True)
@click.option('--systems', multiple=True,
              type=click.Choice(['tersetalk', 'freeform', 'llmlingua', 'hybrid']),
              default=['tersetalk', 'freeform', 'llmlingua'])
@click.option('--n', default=500, help='Examples per task')
@click.option('--seed', default=42)
@click.option('--caps-grid', is_flag=True, help='Run cap ablation grid')
@click.option('--output-dir', default='results/evaluation')
@click.option('--model', default='mistral')
@click.option('--dry-run', is_flag=True, help='Test with 10 examples')
def main(task, systems, n, seed, caps_grid, output_dir, model, dry_run):
    """
    Full evaluation driver for TerseTalk experiments.

    Produces complete results for paper figures.
    """
    # Setup
    set_global_seed(seed)
    output_path = Path(output_dir) / task / f"seed_{seed}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Override n for dry run
    if dry_run:
        n = 10
        print(f"DRY RUN: Using only {n} examples")

    # Load data
    print(f"Loading {n} examples from {task}...")
    if task == 'hotpotqa':
        examples = load_hotpotqa(split='validation', n=n, seed=seed)
    else:
        examples = load_gsm8k(split='test', n=n, seed=seed)

    # Initialize model
    if model == 'echo' or dry_run:
        client = EchoModel()
    else:
        client = ModelClient(model)

    # Define cap configurations for ablation
    if caps_grid:
        cap_configs = [
            {"f": 20, "p": 15, "q": 20, "name": "aggressive"},
            {"f": 30, "p": 20, "q": 30, "name": "baseline"},
            {"f": 50, "p": 40, "q": 50, "name": "relaxed"},
            {"f": 100, "p": 80, "q": 100, "name": "very_relaxed"}
        ]
    else:
        cap_configs = [{"f": 30, "p": 20, "q": 30, "name": "baseline"}]

    # Run experiments
    all_results = {}

    for system in systems:
        print(f"\nRunning {system}...")

        if system == 'tersetalk':
            # Run with each cap configuration
            for cap_config in cap_configs:
                caps = {k: v for k, v in cap_config.items() if k != 'name'}
                config_name = f"{system}_{cap_config['name']}"

                results = run_system_evaluation(
                    examples, client, system,
                    config={'caps': caps}
                )

                all_results[config_name] = results
                save_results(results, output_path / f"{config_name}.jsonl")

        elif system == 'hybrid':
            # Hybrid with different token budgets
            for budget in [400, 600, 800]:
                config_name = f"{system}_budget_{budget}"

                results = run_system_evaluation(
                    examples, client, 'tersetalk',
                    config={
                        'caps': {"f": 30, "p": 20, "q": 30},
                        'hybrid': True,
                        'token_budget': budget
                    }
                )

                all_results[config_name] = results
                save_results(results, output_path / f"{config_name}.jsonl")

        else:
            # Baselines
            results = run_system_evaluation(
                examples, client, system, config={}
            )
            all_results[system] = results
            save_results(results, output_path / f"{system}.jsonl")

    # Save summary
    summary = compute_summary(all_results)
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print_summary_table(summary)

def run_system_evaluation(examples, client, system, config):
    """Run evaluation for one system configuration."""
    results = []

    for example in tqdm(examples):
        if system == 'tersetalk':
            result = run_pipeline_once(example, client, config)
        elif system == 'freeform':
            result = run_freeform_once(example, client)
        elif system == 'llmlingua':
            result = run_llmlingua_once(example, client)
        else:
            raise ValueError(f"Unknown system: {system}")

        results.append(result)

    return results

def save_results(results, filepath):
    """Save results to JSONL file."""
    with open(filepath, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def compute_summary(all_results):
    """Compute summary statistics for all runs."""
    summary = {}

    for config_name, results in all_results.items():
        # Filter successful results
        successful = [r for r in results if r.get('status') != 'error']

        summary[config_name] = {
            'n_total': len(results),
            'n_successful': len(successful),
            'avg_tokens': sum(r['tokens'] for r in successful) / len(successful),
            'accuracy': sum(1 for r in successful if r.get('correct', False)) / len(successful),
            'compliance_rate': len(successful) / len(results)
        }

    return summary

def print_summary_table(summary):
    """Print formatted summary table."""
    print("\n" + "="*60)
    print(f"{'System':<25} {'Tokens':<10} {'Accuracy':<10} {'Compliance':<10}")
    print("-"*60)
    for name, stats in summary.items():
        print(f"{name:<25} {stats['avg_tokens']:<10.1f} "
              f"{stats['accuracy']:<10.2%} {stats['compliance_rate']:<10.2%}")
```

**DoD:** Runs 500 examples; produces cap ablation grid; saves versioned results; generates summary.

#### PR-15 — Analysis Polish & Metrics Provenance (~240 LOC)

Priority: BEFORE EXPERIMENTS
Combining analysis improvements with provenance tracking

```python
# scripts/analyze_v05.py (update existing)

def generate_pareto_curve(results_dir: Path, output_dir: Path):
    """
    Generate quality vs tokens Pareto frontier.

    Objective: minimize tokens, maximize accuracy (multi-objective).
    """
    systems_data = load_all_systems(results_dir)

    # Compute Pareto frontier
    pareto_points = []
    for name, data in systems_data.items():
        point = {
            'system': name,
            'tokens': data['avg_tokens'],
            'accuracy': data['accuracy'],
            'is_pareto': False
        }
        pareto_points.append(point)

    # Mark Pareto-optimal points
    for p1 in pareto_points:
        p1['is_pareto'] = True
        for p2 in pareto_points:
            if p2['tokens'] < p1['tokens'] and p2['accuracy'] >= p1['accuracy']:
                p1['is_pareto'] = False
                break
            elif p2['tokens'] <= p1['tokens'] and p2['accuracy'] > p1['accuracy']:
                p1['is_pareto'] = False
                break

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for point in pareto_points:
        marker = 'o' if point['is_pareto'] else 'x'
        ax.scatter(point['tokens'], point['accuracy'],
                  marker=marker, s=100, label=point['system'])

    # Connect Pareto frontier
    pareto_only = [p for p in pareto_points if p['is_pareto']]
    pareto_only.sort(key=lambda x: x['tokens'])

    if pareto_only:
        xs = [p['tokens'] for p in pareto_only]
        ys = [p['accuracy'] for p in pareto_only]
        ax.plot(xs, ys, 'r--', alpha=0.5, label='Pareto Frontier')

    ax.set_xlabel('Total Tokens (lower is better)')
    ax.set_ylabel('Task Accuracy (higher is better)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_frontier.pdf')
    print(f"Pareto curve saved to {output_dir / 'pareto_frontier.pdf'}")

def generate_ablation_plots(results_dir: Path, output_dir: Path):
    """Generate cap ablation plots with deterministic ordering."""

    # Load cap configurations in deterministic order
    cap_configs = [
        ("aggressive", 20),
        ("baseline", 30),
        ("relaxed", 50),
        ("very_relaxed", 100)
    ]

    results = []
    for name, avg_cap in cap_configs:
        data = load_system_data(results_dir / f"tersetalk_{name}.jsonl")
        if data is None:
            print(f"Warning: No data for tersetalk_{name}, skipping")
            continue

        overflow_rate = data.get('overflow_rate', float('nan'))
        if np.isnan(overflow_rate):
            print(f"Warning: overflow_rate is NaN for {name}")

        results.append({
            'name': name,
            'avg_cap': avg_cap,
            'tokens': data['avg_tokens'],
            'accuracy': data['accuracy'],
            'overflow_rate': overflow_rate
        })

    if not results:
        print("Warning: No cap ablation data found")
        return

    # Plot with provenance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Tokens vs cap size
    ax1.plot([r['avg_cap'] for r in results],
             [r['tokens'] for r in results], 'o-', linewidth=2)
    ax1.set_xlabel('Average Cap Size (tokens)')
    ax1.set_ylabel('Total Tokens Used')
    ax1.grid(True, alpha=0.3)

    # Accuracy vs overflow
    valid_overflow = [r for r in results if not np.isnan(r['overflow_rate'])]
    if valid_overflow:
        ax2.plot([r['overflow_rate'] for r in valid_overflow],
                [r['accuracy'] for r in valid_overflow], 'o-', linewidth=2)
    ax2.set_xlabel('Overflow Rate')
    ax2.set_ylabel('Task Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_caps.pdf')

def enrich_summary_with_provenance(summary_path: Path):
    """Add provenance information to summary.json."""

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Add provenance fields
    for system_name in summary:
        summary[system_name].update({
            'tokens_method': 'tiktoken' if tiktoken_available() else 'heuristic',
            'sp_method': 'bertscore' if bertscore_available() else 'jaccard',
            'timestamp': datetime.now().isoformat(),
            'version': get_repo_version()  # Git hash if available
        })

    # Save enriched summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

def tiktoken_available():
    """Check if tiktoken is available."""
    try:
        import tiktoken
        return True
    except ImportError:
        return False

def bertscore_available():
    """Check if BERTScore is available."""
    try:
        import bert_score
        return True
    except ImportError:
        return False
```

**DoD:** Pareto frontier identified correctly; ablation plots deterministic; provenance tracked.

#### PR-16 — Statistical Significance Testing (~200 LOC)

```python
# tersetalk/statistics.py (new file)

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple

def bootstrap_ci(data1: List[float], data2: List[float],
                 n_bootstrap: int = 10000,
                 confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for difference.

    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    diffs = []
    n1, n2 = len(data1), len(data2)

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(data1, n1, replace=True)
        sample2 = np.random.choice(data2, n2, replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))

    # Compute percentile CI
    alpha = 1 - confidence
    ci_lower = np.percentile(diffs, 100 * alpha/2)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha/2))
    mean_diff = np.mean(data1) - np.mean(data2)

    return mean_diff, ci_lower, ci_upper

def test_noninferiority(treatment: List[float], control: List[float],
                       delta: float = 0.02,
                       confidence: float = 0.95) -> Dict:
    """
    Test if treatment is non-inferior to control.

    H0: treatment - control < -delta (treatment is inferior)
    H1: treatment - control >= -delta (treatment is non-inferior)

    Args:
        treatment: Accuracy values for treatment (e.g., Hybrid)
        control: Accuracy values for control (e.g., LLMLingua)
        delta: Non-inferiority margin (default 2%)
        confidence: Confidence level (default 95%)

    Returns:
        Dictionary with test results
    """
    mean_diff, ci_lower, ci_upper = bootstrap_ci(treatment, control, confidence=confidence)

    # For non-inferiority, check if lower bound > -delta
    is_noninferior = ci_lower > -delta

    return {
        'mean_difference': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'delta': delta,
        'is_noninferior': is_noninferior,
        'p_value_approx': 1 - confidence if is_noninferior else confidence
    }

def paired_t_test(before: List[float], after: List[float]) -> Dict:
    """
    Paired t-test for token reduction claims.

    Use when same examples are run through different systems.
    """
    diffs = [a - b for a, b in zip(after, before)]
    t_stat, p_value = stats.ttest_1samp(diffs, 0)

    return {
        'mean_reduction': np.mean(diffs),
        'std_reduction': np.std(diffs),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def compute_all_significance_tests(results_dir: Path) -> Dict:
    """
    Run all significance tests for paper claims.
    """
    tests = {}

    # Load results
    tersetalk = load_accuracies(results_dir / "tersetalk_baseline.jsonl")
    freeform = load_accuracies(results_dir / "freeform.jsonl")
    llmlingua = load_accuracies(results_dir / "llmlingua.jsonl")
    hybrid = load_accuracies(results_dir / "hybrid_budget_600.jsonl")

    # Test 1: Token reduction significance
    tersetalk_tokens = load_tokens(results_dir / "tersetalk_baseline.jsonl")
    freeform_tokens = load_tokens(results_dir / "freeform.jsonl")

    tests['token_reduction'] = paired_t_test(freeform_tokens, tersetalk_tokens)

    # Test 2: Quality preservation
    tests['quality_preservation'] = bootstrap_ci(tersetalk, freeform)

    # Test 3: Hybrid non-inferiority
    tests['hybrid_noninferiority'] = test_noninferiority(
        hybrid, llmlingua, delta=0.02
    )

    return tests
```

##### scripts/run_significance.py (new file)

```python
@click.command()
@click.option('--results-dir', required=True, type=Path)
@click.option('--output', default='significance_tests.json')
def main(results_dir, output):
    """Run statistical significance tests for paper claims."""

    tests = compute_all_significance_tests(results_dir)

    # Print summary
    print("\n" + "="*60)
    print("SIGNIFICANCE TEST RESULTS")
    print("="*60)

    # Token reduction
    tr = tests['token_reduction']
    print(f"\nToken Reduction: {tr['mean_reduction']:.1%} "
          f"(p={tr['p_value']:.4f}, {'SIG' if tr['significant'] else 'NS'})")

    # Quality preservation
    qp = tests['quality_preservation']
    print(f"\nQuality Difference: {qp[0]:.3f} "
          f"[95% CI: {qp[1]:.3f}, {qp[2]:.3f}]")

    # Hybrid non-inferiority
    hi = tests['hybrid_noninferiority']
    print(f"\nHybrid Non-Inferiority: {'PASS' if hi['is_noninferior'] else 'FAIL'}")
    print(f"  Difference: {hi['mean_difference']:.3f}")
    print(f"  95% CI: [{hi['ci_lower']:.3f}, {hi['ci_upper']:.3f}]")
    print(f"  Margin: {hi['delta']:.3f}")

    # Save full results
    with open(output, 'w') as f:
        json.dump(tests, f, indent=2)
```

**DoD:** All major claims have p-values; non-inferiority tested; results in paper-ready format.

### 4.5 Week 7: Optional Topology Extension

#### PR-V1 — Binary Topology Router (~350 LOC)

**Context:** Simple overflow-triggered topology switching.

**Public API (`tersetalk/topology_router.py`):**

```python
class BinaryTopologyRouter:
    def __init__(self, overflow_threshold=0.3):
        self.threshold = overflow_threshold

    def select_topology(self, initial_msg: str, overflow_stats: dict) -> str:
        """
        Select topology based on overflow frequency.
        High overflow → needs more coordination → Collaborative
        Low overflow → simple enough → Direct
        """
        overflow_rate = overflow_stats.get('rate', 0.0)

        if overflow_rate >= self.threshold:
            return 'collaborative'  # Chain+Verify
        else:
            return 'direct'  # Manager-coordinated

    def run_collaborative(self, example: dict, client: ModelClient) -> dict:
        """
        Manager → Worker₁ → Worker₂ → Critic → Manager → Worker₁
        Single verify/fix loop for complex tasks
        """
        # Implementation similar to pipeline but with chain flow
```

**DoD:** Router selects correctly; both topologies execute.

### 4.6 Measurement & Monitoring

#### Lightweight Performance Monitoring

```python
class LightweightMonitor:
    """Minimal overhead performance tracking"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.metrics = defaultdict(list)

    @contextmanager
    def timer(self, name: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.metrics[name].append(elapsed)

    def log_async(self):
        """Non-blocking logging to avoid I/O overhead"""
        threading.Thread(target=self._write_metrics).start()
```

---

## 5) Dependencies

### Required (CORE)

```
instructor>=1.0.0      # For structured generation
pydantic>=2.0.0        # For data validation and models
openai>=1.3.0          # Client library for Instructor (works with Ollama)
ollama>=0.1.0          # To manage local model server
datasets>=2.0.0             # HotpotQA, GSM8K
llmlingua==0.2.2           # Baseline compression
pytest>=7.0.0              # Testing
pytest-timeout>=2.1.0      # CI stability
click>=8.0.0               # CLI framework
tqdm>=4.0.0                # Progress bars
matplotlib>=3.5.0          # Analysis plots
```

### Optional (auto-disabled if missing)

```
bert-score>=0.3.0          # Semantic preservation
torch>=1.9.0               # For BERTScore
tiktoken>=0.3.0            # Exact token counting
simdjson>=5.0.0            # Fast JSON parsing
msgpack>=1.0.0             # Protocol comparison
protobuf>=3.0.0            # Protocol comparison
```

---

## 6) Risk Analysis & Mitigations

### Technical Risks

**Risk 1: Structured generation fails or produces low-quality output**

- **Mitigation:** Instructor has built-in retry logic. Pydantic models can be refined. Report failure rates as a key finding.
- **Measurement:** Track compliance rate, report as finding
- **Narrative:** "Identifying model limitations is a contribution"

**Risk 2: Memory overflow handling adds complexity**

- **Mitigation:** Simple extractive summarization as default
- **Measurement:** Ablation comparing summary methods
- **Narrative:** "Protocol works with any summarization backend"

**Risk 3: Topology switching shows no benefit**

- **Mitigation:** Focus paper on protocol-only contributions
- **Measurement:** Report negative result with analysis
- **Narrative:** "Protocol enables future topology optimization"

### Schedule Risks

**Risk 4: Implementation delays**

- **Mitigation:** Weekly milestones with go/no-go decisions
- **Week 1 checkpoint:** Structured generation with Instructor working? If no → simplify Pydantic models or debug prompts.
- **Week 3 checkpoint:** Microbenchmarks show gains? If no → adjust claims
- **Week 5 checkpoint:** Evaluation scaling? If no → reduce N

**Risk 5: Protocol under-compresses vs LLMLingua**

- **Mitigation:** Hybrid gate auto-routes the turn to Free-form+LLMLingua when projected tokens exceed budget; LLMLingua also compresses long tag payloads and deref text.
- **Measurement:** log gate decisions; report % routed; show Pareto including Hybrid curve.
- **Acceptance guard:** if token saving <20% on smoke set with gate off, re-run with gate on in CI.

---

## 7) Evaluation Plan

### Primary Experiments

**Experiment 1: Token Efficiency**

- **Setup:** 500 examples each from HotpotQA, GSM8K
- **Systems:** TerseTalk, Free-form, Free-form+LLMLingua
- **Metrics:** Total tokens, quality (EM/exact match)
- **Expected:** 25-35% reduction, <2% quality loss

**Experiment 2: Systems Performance**

- **Setup:** Same 500 examples
- **Metrics:** TPS, TTFT, ITL, serialization overhead
- **Expected:** Measurable latency reduction due to fewer tokens and faster (de)serialization.

**Experiment 3: Microbenchmarks**

- **Setup:** Synthetic workloads
- **Metrics:** Tag extraction speed, boundary detection, memory lookup
- **Expected:** ≥10× improvement on at least one

**Experiment 4: Hybrid Non-Inferiority (quality)**

- **Setup:** same 500-example shards.
- **Systems:** add Hybrid (gate on).
- **Test:** one-sided non-inferiority (δ = 0.02 absolute) of Hybrid vs LLMLingua on accuracy; if lower-bound 95% CI of (Hybrid − LLMLingua) > −δ ⇒ non-inferior.

### Ablation Studies

**Ablation 1: Cap Sensitivity**

```python
cap_configs = [
    {"f": 20, "p": 15, "q": 20},   # Aggressive
    {"f": 30, "p": 20, "q": 30},   # Baseline
    {"f": 50, "p": 40, "q": 50},   # Relaxed
    {"f": 100, "p": 80, "q": 100}  # Very relaxed
]
```

**Ablation 2: Summarization Methods**

- Extractive (default)
- LLMLingua
- Simple truncation

**Ablation 3: Enforcement Mechanism Impact (Instructor+Pydantic vs JSON mode vs llama.cpp+GBNF ablation).**

- Instructor with Pydantic: (Default) Robust validation and automatic retries.

- Ollama JSON Mode: Use the model's native JSON output mode without Pydantic's rich validation. This tests if the stricter schema and retry logic add value.

**Ablation 4: Model Sensitivity** (if time)

- Mistral-7B (primary)
- Llama-3.1-8B
- Qwen2.5-14B

**Ablation 5:**

- Gate on/off and token budget ∈ {400, 600, 800}.

**Ablation 6:**

- Summarizer: extractive vs LLMLingua vs truncate.

**Ablation 7:**

-Dereference policy: always / conditional / never.

**Ablation 8:**

-Protocol only vs Hybrid on hard-length bins.

---

## 8) Paper Positioning

### Title Options

1. "TerseTalk: A Typed JSONL Protocol for Token-Efficient and Reproducible Multi-Agent Communication"
2. "Less is More: A Systems Approach to Efficient Multi-Agent LLM Communication with Typed Protocols"
3. "From Free-form to Fast: A Systems Approach to Multi-Agent Communication"

### Abstract Structure (150 words)

```
Multi-agent LLM systems waste tokens on verbose, free-form messages.
We present TerseTalk-JSONL, a typed protocol that makes this communication efficient and reliable.
Our contributions are: (1) a compact JSONL protocol with overflow pointers (`M#`) to prevent information loss; (2) a reproducible implementation using validation-driven generation (`Instructor` + `Pydantic`); and (3) a demonstration of 10× faster message parsing.
On HotpotQA and GSM8K, TerseTalk achieves a 25-35% token reduction with <2% quality loss compared to baselines. Our work provides a practical, open-source system for building robust multi-agent applications.
```

### Key Claims

1. **"A typed protocol is the key to efficient multi-agent communication, independent of the generation stack"**
2. **"Typed protocols enable 10× faster message processing"**
3. **"Validation-driven generation with tools like Instructor provides 100% protocol compliance with superior reproducibility and lower implementation overhead"**

---

## 9) Appendices

### A. Glossary

**Terms:**

- **JSONL/NDJSON:** One JSON value per line, streaming-friendly format
- **M# references:** Memory pointers for overflowed content (like footnotes)
- **Soft caps:** Target lengths triggering overflow, not hard truncation
- **Structured Generation:** Using libraries like Instructor with schemas (e.g., Pydantic models) to guarantee valid, typed model outputs.
- **DoD:** Definition of Done - acceptance criteria for each PR
- **TPS/TTFT/ITL:** Tokens per second, Time to first token, Inter-token latency
- **SP:** Semantic preservation - how much meaning is retained
- **Manager-coordinated pipeline:** Our M→W→C flow (not true star topology)

### B. Quick Command Reference

```bash
# Install dependencies
make install

# Run echo tests (no GPU)
MODEL_CLIENT=echo make test

# Run small experiment
python scripts/run_v05.py \
    --task hotpotqa \
    --system tersetalk \
    --n 10 \
    --seed 42 \

# Run full evaluation
python scripts/run_v05.py \
    --task hotpotqa \
    --system tersetalk \
    --n 500 \
    --seed 42 \
    --caps '{"f":30,"p":20,"q":30}' \
    --model mistral:instruct

# Generate figures
python scripts/analyze_v05.py \
    --indir results \
    --outdir figures

# Run microbenchmarks
python benchmarks/run_all.py

# Clean old results
python scripts/cleanup.py --keep-last 5

# Calibrate best caps/policies for a task (writes configs/calibration.yaml)
python scripts/calibrate_caps.py --task hotpotqa --n 50 --seed 0

# Run with hybrid gate and explicit token budget
python scripts/run_v05.py \
  --task hotpotqa --system tersetalk --n 500 --seed 42 \
  --hybrid --token-budget 600 --deref-policy conditional \
  --summarizer llmlingua

# Turn the LLMLingua touchpoints on/off independently
# (flags: --preoverflow-ll2, --overflow-ll2, --deref-ll2)
python scripts/run_v05.py --preoverflow-ll2 --overflow-ll2 --deref-ll2 ...
```

### C. Model Setup Guide

```bash
# 1. Install Ollama
# Follow instructions at https://ollama.com

# 2. Pull the Mistral 7B model
ollama pull mistral

# 3. Run the Ollama server (it may already be running as a service)
ollama serve

# 4. Test with a simple curl command (or the Python client)
curl http://localhost:11434/v1/chat/completions -d '{
"model": "mistral",
"messages": [{"role": "user", "content": "Why is the sky blue?"}],
"stream": false
}'
```

---

## Timeline Summary

**Week 1-2:** Protocol + Structured Generation (MUST work)
**Week 3-4:** Microbenchmarks (MUST show 10× somewhere)
**Week 5-6:** Full evaluation (MUST show 25% reduction)
**Week 7:** Topology (ONLY if ahead)
**Week 8:** Writing (focus on systems narrative)

**Go/No-Go Decisions:**

- End of Week 1: Grammar working? If no → pivot to post-hoc only
- End of Week 3: Microbenchmarks promising? If no → adjust claims
- End of Week 5: On track? If no → skip topology entirely

**Success Criteria:**

- Validation-driven generation ensures 100% protocol compliance, while the typed, single-letter tag design enables 10× faster message parsing compared to free-form approaches.
- At least one 10× microbenchmark win
- 25-35% token reduction with <2% quality loss
- Reproducible with fixed seeds
- Clean, documented code with tests
- Hybrid is non-inferior to LLMLingua on accuracy at δ=2% (95% one-sided CI).
- Gate maintains ≥20% token saving on smoke set (else auto-rollback or stricter config via CI).

---

## Final Checklist Before Starting

- [ ] Download Mistral model
- [ ] Install Ollama and pull the baseline model (e.g., ollama pull mistral)
- [ ] Verify Instructor can connect to the Ollama server
- [ ] Write a simple Pydantic model and test structured generation
- [ ] Set up results directory structure
- [ ] Initialize git repo with .gitignore
- [ ] Create requirements.txt
- [ ] Write initial README
- [ ] Set up pytest with first dummy test
- [ ] Configure VSCode/IDE with Python environment
- [ ] Block calendar for focused coding time

**Start Date:** \***\*\_\_\_\*\***
**Target Submission:** MLSys 2026 (2 months)

---

_This proposal represents a systems-first approach to multi-agent communication, emphasizing measurable performance improvements and practical implementation over theoretical contributions._

# 16) Extension (v1.0): Binary Task-Adaptive Topology (scoped)

> Keep v0.5 unchanged as the core result. v1.0 adds a **minimal online topology switcher** that leverages TerseTalk-JSONL signals. We position this as **online, instance-adaptive** (per example), contrasting with **offline topology design** like **G-Designer**.

## v1.0 goals (narrow scope)

- **Topologies:** support **Direct Mode (Star)** and **Collaborative Mode (Chain+Verify)**.
- **Router:** simple **binary decision** using typed signals (counts of `f/u/p/q`, presence of `d:M#` and code fences).
- **Metrics:** topology selection accuracy (vs task labels), token/latency, overflow rate, quality.
- **Safety valves:** cap switch frequency (min-dwell 1 round), cap fan-out.

### Benchmarks suited for topology differences

We evaluate inside **MARBLE (MultiAgentBench)** and staple tasks where the **optimal topology** is intuitive:  
**Direct-optimal**: **BoolQ**, **CommonsenseQA**, **SocialIQA**.  
**Collaborative-optimal**: **HotpotQA**, **GSM8K**, **HumanEval**.  
(Optionally add SQuAD for reading-comprehension variety.)

### Core design: two topologies + one explainable router

**Topologies**

- **Direct Mode (Star):** Manager → Worker → Critic (good for single-path Q/A)
- **Collaborative Mode (Chain+Verify):** Manager → Worker₁ → Worker₂ → Critic → Manager → Worker₁ (one verify/fix loop; good for multi-step/code)

**Router using TerseTalk-JSONL signals** (≈200 LOC in `PR-V1.1`)

````python
# tersetalk/topology_router.py
class BinaryAdaptiveRouter:
    def __init__(self, complexity_threshold=3.0):
        self.threshold = complexity_threshold

    def analyze_complexity(self, jsonl_lines):
        signals = {
            "facts":       sum(1 for l in jsonl_lines if l[0] == "f"),
            "plans":       sum(1 for l in jsonl_lines if l[0] == "p"),
            "questions":   sum(1 for l in jsonl_lines if l[0] == "q"),
            "assumptions": sum(1 for l in jsonl_lines if l[0] == "u"),
            "has_refs":    any(l[0] == "d" for l in jsonl_lines),
            "has_code":    any("```" in str(l) for l in jsonl_lines),
        }
        complexity = (
            signals["facts"] * 1.0 +
            signals["plans"] * 2.0 +
            signals["questions"] * 1.5 +
            (3.0 if signals["has_refs"] else 0.0) +
            (4.0 if signals["has_code"] else 0.0)
        )
        return complexity, signals

    def select_topology(self, initial_msg_jsonl):
        complexity, signals = self.analyze_complexity(initial_msg_jsonl)
        return ("collaborative" if complexity >= self.threshold else "direct"), signals
````

**Binary decision flow**

```mermaid
flowchart LR
  A[Manager JSONL<br/>r,g,f,u,p,q,d] --> B{{Complexity ≥ θ?}}
  B -- "No"  --> S[Direct Mode<br/>Star: M→W→C]
  B -- "Yes" --> C[Collaborative Mode<br/>Chain+Verify]
  C --> L1[ M→W1→W2→C ]
  L1 --> L2[ C→M (critique) ]
  L2 --> L3[ M→W1 (fix) ]
```

**Collaborative (Chain+Verify) message flow**

```mermaid
sequenceDiagram
  participant M as Manager (JSONL)
  participant W1 as Worker₁ (reason)
  participant W2 as Worker₂ (refine/code)
  participant C as Critic (verify)
  M->>W1: subgoal+facts (r,g,f,u,p)
  W1->>W2: draft + plan (p,f; d: M#)
  W2->>C: proposed answer/code
  C-->>M: verdict ["v","A|R|E"] + notes
  M->>W1: one fix pass (if "R")
```

### MARBLE experimental protocol (balanced, auditable)

**Task selection (600 total; 100 per task)**

```python
MARBLE_TASKS = {
  "direct_optimal":      ["BoolQ", "CommonsenseQA", "SIQA"],
  "collaborative_optimal": ["HotpotQA", "GSM8K", "HumanEval"]
}
```

**Systems to compare**

- Fixed-Star (always Star)
- Fixed-Chain (always Chain+Verify)
- Random-Switch (50/50)
- Oracle (task-label topology; upper bound)
- FreeForm-Adaptive (same binary rules on free-form messages)
- **TerseTalk-Adaptive (ours)** (binary rules on typed protocol)

**Metrics**

```python
selection_accuracy = correct_topology_choices / total_choices
oracle_agreement  = agreement_with_task_labels / total_tasks

token_reduction      = (tokens_fixed_best - tokens_adaptive) / tokens_fixed_best
quality_preservation = accuracy_adaptive / accuracy_fixed_best
efficiency_score     = token_reduction * quality_preservation

protocol_advantage = terseTalk_selection_accuracy / freeform_selection_accuracy
```

**Targets (not claims):**

- Topology selection ≥ **85%** agreement with task labels (oracle).
- **20–30%** token reduction vs best fixed topology with ≤ **2%** quality drop.
- Protocol advantage: ~**2×** better selection accuracy vs free-form (typed fields expose `p/f/q/d` cleanly).

### Implementation plan (2 weeks additional)

**Week v1.1 — Router + collaborative executor**

- **PR-V1.1: `topology_router.py` (~200 LOC)** — binary router (above); tests: synthetic JSONL → topology; threshold monotonicity.
- **Add `run_collaborative_once(...)` (~150 LOC)** — single verify/fix loop; EchoModel test; metrics match `run_star_once` (+signals).

**Week v1.2 — MARBLE runner**

- **PR-V1.2: `scripts/run_marble_v1.py` (~250 LOC)** — run all 6 systems across 6 tasks, N=100 each; outputs per-system CSVs + summary JSON.
- **Ablations:** θ ∈ {1,3,5,7}; ablate `{facts,plans,questions,has_refs,has_code}`; protocol necessity (scrambled tags); leave-one-task-out.
- **Failure-mode checks:** phase/session tags enforce ordering; deref depth/τ guard; min-dwell=1 round to avoid thrash.

---

## 17) Optional v1.1: Density-triggered router & sparse mesh (if time allows)

- **Density-Adaptive Router** — use protocol **information density** slope to choose between star/chain/mesh.
- **Adaptive Mesh (sparse)** — prune a complete graph using **protocol affinity** (e.g., similar tag patterns), ensure connectivity; single round only.
- **Metrics:** add **routing decision ms**, **switch latency**, **avg path length**.

> These are compelling but **out of scope** for the initial v1.0 binary target; include only if schedule permits.

---

## 18) Deployment (optional, nice-to-have for artifact badge)

Provide a simple container setup so others can run echo-based tests and CPU microbenchmarks.

**PR-D-01 — docker-compose (≤120 LOC)**

```yaml
version: '3.9'
services:
  tersetalk-router:
    image: tersetalk:latest
    build: .
    environment:
      GRAMMAR_ENGINE: 'none' # or "llama_cpp" / "guidance"
      TOPOLOGY_MODE: 'direct' # or "adaptive"
    deploy:
      resources:
        limits:
          memory: 4G
```

**PR-D-02 — Makefile helpers (≤80 LOC)**  
Targets: `install`, `test`, `run-synth`, `bench`.

---

## Appendix A — Glossary (with plain-English tie-ins)

- **JSONL / NDJSON.** “One JSON value per line”; streaming-friendly and easy to validate. (Each line is an independent record.)
- **Soft cap.** A target length per field; try to compress; if still too long, emit an `["o", summary, "M#id"]` overflow line. (Don’t truncate meaning.)
- **M# memory.** Bounded key-value store for overflowed text; entries referenced by IDs like `M#23`. (Like footnotes.)
- **Semantic preservation (SP).** How much meaning the typed message retains vs. the free-form baseline (BERTScore/BLEURT). (Does the compressed message still say the same thing?)
- **Star / Chain / Flat.** Canonical agent topologies for message passing. (Star = hub-and-spoke; Chain = pipeline; Flat = everyone talks in one round.)
- **Topology switching.** Choosing a topology per instance, online. (Flip between Star/Chain/Flat based on simple signals.)
- **Prompt compression.** Techniques like LLMLingua/-2 that shorten prompts while trying to preserve meaning.
- **Structured decoding.** Enforcing output shapes (JSON schema/CFG) during generation (e.g., vLLM or llama.cpp grammars). (v0.5 is post-hoc; v1.x can enable at generation.)
- **TPS / TTFT / ITL.** Throughput (tokens/s), Time-to-First-Token, and Inter-Token Latency (per-token pace after TTFT).

---

## One-slide TL;DR (for your PI)

- **v0.5 = protocol-only:** TerseTalk-JSONL (typed, compact, soft-capped, overflow pointers).
- **Baselines:** free-form and free-form + LLMLingua/-2.
- **Tight PRs (≤250 LOC)**, GPU-free CI via `EchoModel`.
- **Metrics:** tokens, EM/Acc, SP (BERTScore), overflow/memory, JSONL serde latency (optional TPS/TTFT/ITL).
- **v1.0 extension:** binary topology switching (Direct vs Collaborative) guided by TerseTalk signals, evaluated in MARBLE; selection/efficiency metrics; compares vs fixed and random baselines.
- **Optionals that strengthen MLSys submission:** grammar-constrained JSONL, streaming processing, microbenchmarks, industrial protocol baselines, and density-aware routing.
- **Positioning:** protocol–topology **co-design**; complements offline topology design (G-Designer) and message compression (LLMLingua).

---

## (Optional) Queueing theory context

If capacity allows, add a light queueing-style utilization backdrop (M/M/c) to reason about expected wait vs. fan-out across Star/Chain. Treat as qualitative context only; do **not** overclaim.
