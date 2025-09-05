# TerseTalk: A Typed JSONL Protocol for Token‑Efficient Multi‑Agent LLMs (v0.5) **and** a Scoped Topology‑Adaptive Extension (v1.0)

> **Final Version with All Updates** — Conditional Approval & Revised 8‑Week MLSys Plan  
> **Status:** ✅ APPROVED **with conditions**. Focus on **protocol + microbenchmarks** first; topology is **bonus**.  
> _Last updated: 2025-09-04_

## 0) Executive Summary

### Problem gap

Most multi‑agent LLM systems pass **free‑form natural language** between agents—verbose, costly in tokens, and brittle to parse. Prompt compression (e.g., **LLMLingua/‑2**) reduces tokens, but treats messages as generic text rather than an **on‑wire, typed inter‑agent protocol** ([LLMLingua](https://arxiv.org/abs/2310.05736), [LLMLingua‑2](https://arxiv.org/abs/2403.12968)).

### Our v0.5 proposal (protocol‑only)

We introduce **TerseTalk‑JSONL**: a compact, newline‑delimited JSON protocol (**JSON Lines/NDJSON**) with **single‑letter tags**, **soft token caps**, and **overflow pointers** (`M#`) into a bounded memory store. We evaluate against **Free‑form** and **Free‑form + LLMLingua/‑2** on **HotpotQA** (multi‑hop QA) and **GSM8K** (multi‑step math) with metrics covering:

- **Quality vs. tokens (Pareto)**
- **Semantic preservation (BERTScore; BLEURT optional)**
- **Overflow & memory behavior** (reset per task)
- **(De)serialization latency & bytes‑on‑wire**
- **Throughput & latency (TPS, TTFT, ITL)**

**Core Innovation:** Grammar‑constrained generation via llama-cpp makes structured output **faster, not slower**, while guaranteeing protocol compliance.

### Why now (and why JSONL)

- **JSON Lines/NDJSON**: standard, streaming‑friendly "one JSON value per line"; easy to validate/diff/pipe; **O(1) message boundary detection**
- **Typed lines** drop verbose keys and enforce concise structure, enabling **10× faster tag extraction**
- **Soft caps + overflow pointers** avoid catastrophic truncation while retaining full content via `M#` dereference (like footnotes)
- **Grammar constraints** reduce TTFT by 20-30% while ensuring 100% protocol compliance
- **Strong baselines**: LLMLingua/‑2 for fair comparison against state‑of‑the‑art compression

References: [JSON Lines](https://jsonlines.org/), [NDJSON spec](https://github.com/ndjson/ndjson-spec), [HotpotQA](https://aclanthology.org/D18-1259/), [GSM8K](https://arxiv.org/abs/2110.14168), [BERTScore](https://arxiv.org/abs/1904.09675), [BLEURT](https://aclanthology.org/2020.acl-main.704.pdf).

---

## CA. Conditional Approval Requirements (MUST‑Have)

### CA.1 Critical Path

**Weeks 1–2 — Protocol Foundation**

- **Grammar constraints (CORE)**: Get JSONL grammar working with **one GGUF model** (Qwen2.5‑14B‑Instruct or Mistral‑7B)
- **Density proxy**: Use **overflow frequency** as initial metric: `density = 1.0 - (overflow_lines / total_lines)`
- **Mixed‑format detection**: Implement guardrails for format breaks
- **Target:** >90% protocol compliance on 100‑sample set

**Weeks 3–4 — Microbenchmarks (CORE)**

- **MB‑1**: ≥10× faster tag extraction from JSONL vs free‑form
- **MB‑2**: ≥5× faster streaming boundary detection
- **MB‑3**: 20–30% TTFT reduction with grammar constraints

**Weeks 5–6 — Full Evaluation**

- Scale to ≥500 examples per task
- Show 25–35% token reduction with <2% quality loss
- Complete failure analysis

**Week 7 — Topology (Only if Ahead)**

- Implement overflow‑frequency‑triggered binary switch
- Demonstrate on HotpotQA & GSM8K only

**Week 8 — Paper Writing**

### CA.2 Non‑Negotiable Requirements

1. **One baseline GGUF model** (Qwen2.5‑14B‑Instruct‑GGUF Q4_K_M preferred)
2. **Mixed‑format detection** with deterministic fallback
3. **Density calibration** on 50 samples
4. **Reproducibility infrastructure** with global seed control
5. **Memory reset per task** (no state leakage)
6. **Results organization** with versioning and cleanup

### CA.3 Risk Mitigations

- **Grammar fails:** Pivot to post‑hoc correction, report overhead as contribution
- **Topology shows no benefit:** Frame as "protocol enables future optimization"
- **Time shortage:** Submit v0.5 only as 4‑page short paper

---

## 1) What's new vs. prior art (gap analysis)

- **Typed protocol for inter‑agent communication:** Most frameworks (AutoGen, CAMEL) pass free‑form text. TerseTalk defines a **compact, typed on‑wire protocol** designed for **minimal tokens** and **fast parseability**.  
  References: [AutoGen](https://arxiv.org/abs/2308.08155), [CAMEL](https://arxiv.org/abs/2303.17760).

- **Head‑to‑head vs. learned compression:** LLMLingua/‑2 is message‑agnostic; we test **typed protocol vs. learned compression** directly in inter‑agent settings.  
  References: [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/), [LLMLingua‑2](https://arxiv.org/abs/2403.12968).

- **Grammar‑constrained generation as performance win:** Recent benchmarks show constraints can **speed up** generation (Guidance: 50% faster). We're first to apply this to multi‑agent protocols.

- **Systems‑first evaluation:** Token accounting, task quality, semantic preservation, **and** systems metrics (TPS, TTFT, ITL, serialization overhead).

**Takeaway:** v0.5 is a **protocol‑only systems contribution** with measurable wins; v1.0 adds minimal topology adaptation.

---

## 2) TerseTalk‑JSONL Protocol Specification

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

- **Topology:** Manager‑coordinated pipeline (M→W→C flow, not true star)
- **Protocol:** TerseTalk‑JSONL with soft caps and overflow
- **Memory:** Bounded store with `M#` references (reset per task)
- **Grammar:** CORE llama-cpp constraints for guaranteed compliance
- **Baselines:** Free‑form, Free‑form + LLMLingua‑2

### 3.2 Metrics Framework

**Task Performance:**

- HotpotQA: Exact Match (EM)
- GSM8K: Exact answer match
- Token reduction: % saved vs. free‑form
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

---

## 4) Detailed Implementation Plan

### 4.1 Week 1-2: Protocol Foundation

#### PR‑00 — Reproducibility Infrastructure (~150 LOC) [NEW]

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

#### PR‑01 — Repository Scaffold (~80 LOC)

**Before→After:** Empty → working skeleton
**Files:** `requirements.txt`, `README.md`, `Makefile`, `.gitignore`
**DoD:** `make install` works; `python scripts/run_v05.py --help` prints

#### PR‑02 — JSONL Protocol & Validator (~220 LOC)

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

#### PR‑02G — Grammar‑Constrained Generation (CORE) (~230 LOC)

**Context:** Force valid JSONL output, reduce TTFT.

**Public API (`tersetalk/constrained_generator.py`):**

```python
class ConstrainedJSONLGenerator:
    JSONL_GRAMMAR = r'''
    root ::= line+
    line ::= array "\n"
    array ::= "[" ws tag ws ("," ws value ws)* "]"
    tag ::= "\"r\"" | "\"g\"" | "\"f\"" | "\"u\"" | "\"p\"" |
            "\"q\"" | "\"d\"" | "\"v\"" | "\"o\"" | "\"t\"" | "\"x\""
    value ::= string | ref
    string ::= "\"" ([^"\\] | "\\" .)* "\""
    ref ::= "\"M#" [0-9]+ "\""
    ws ::= [ \t]*
    '''

    def __init__(self, model_path: str):
        self.model = llama_cpp.Llama(model_path)
        self.grammar = llama_cpp.LlamaGrammar.from_string(self.JSONL_GRAMMAR)

    def generate(self, prompt: str, max_tokens=256, fallback=True):
        """Generate with grammar constraints"""
        try:
            return self.model.create_completion(
                prompt,
                grammar=self.grammar,
                max_tokens=max_tokens,
                temperature=0.0
            )
        except Exception as e:
            if fallback:
                return self.generate_unconstrained(prompt, max_tokens)
            raise
```

**DoD:** >90% compliance rate; TTFT measurement shows improvement.

#### PR‑03 — Memory Store (~140 LOC)

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

#### PR‑04 — Summarization Module (~180 LOC)

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

### 4.2 Week 3-4: Microbenchmarks (CORE)

#### PR‑MB — Microbenchmark Suite (~600 LOC total)

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

# benchmarks/grammar_performance.py
def benchmark_grammar_ttft():
    """TTFT with/without grammar constraints"""
```

**DoD:**

- Tag extraction: ≥10× speedup
- Streaming: ≥5× speedup
- Grammar: 20-30% TTFT reduction (or documented failure analysis)

### 4.3 Week 5-6: Full System Implementation

#### PR‑05 — Model I/O (~220 LOC)

**Public API (`tersetalk/model_io.py`):**

```python
class ModelClient:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.generator = None

    def call_jsonl_strict(self, system: str, user_jsonl: str, max_tokens=256) -> str:
        """Generate response with JSONL constraints"""
        if not self.generator:
            self.generator = ConstrainedJSONLGenerator(self.model_path)

        prompt = f"{system}\n\nUser JSONL:\n{user_jsonl}\n\nAssistant JSONL:"
        response = self.generator.generate(prompt, max_tokens)

        # Validate output
        validator = JSONLValidator({}, MemoryStore())
        is_mixed, break_line = validator.detect_format_break(response)
        if is_mixed:
            raise ValueError(f"Format break at line {break_line}")

        return response

class EchoModel(ModelClient):
    """For CI testing without GPU"""
    def call_jsonl_strict(self, system: str, user_jsonl: str, max_tokens=256) -> str:
        return user_jsonl  # Echo back for testing
```

**DoD:** Strict JSONL enforced; Echo enables GPU-free CI.

#### PR‑06 — Dataset Adapters (~200 LOC)

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

#### PR‑07 — Manager-Coordinated Pipeline (~230 LOC)

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

#### PR‑08 — Baselines (~230 LOC)

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

#### PR‑09 — Results Manager (~180 LOC) [NEW]

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

#### PR‑10 — Metrics Module (~240 LOC)

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

**DoD:** All metrics compute correctly; fallbacks work.

#### PR‑11 — Experiment Driver (~250 LOC)

**Public API (`scripts/run_v05.py`):**

```python
@click.command()
@click.option('--task', type=click.Choice(['hotpotqa', 'gsm8k']))
@click.option('--system', type=click.Choice(['tersetalk', 'freeform', 'llmlingua']))
@click.option('--n', default=100, help='Number of examples')
@click.option('--seed', default=0, help='Random seed')
@click.option('--caps', default='{"f":30,"p":20,"q":30}', help='Soft caps (JSON)')
@click.option('--grammar/--no-grammar', default=True)
@click.option('--model', default='qwen2.5-14b-instruct-q4_k_m.gguf')
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
        'grammar': grammar,
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

#### PR‑12 — Analysis Scripts (~220 LOC)

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
    """Ablation studies on caps, grammar, etc."""
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

### 4.5 Week 7: Optional Topology Extension

#### PR‑V1 — Binary Topology Router (~350 LOC)

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
llama-cpp-python>=0.2.0    # Grammar-constrained generation
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

**Risk 1: Grammar constraints fail with chosen model**

- **Mitigation:** Fall back to post-hoc validation
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
- **Week 1 checkpoint:** Grammar working? If no → pivot
- **Week 3 checkpoint:** Microbenchmarks show gains? If no → adjust claims
- **Week 5 checkpoint:** Evaluation scaling? If no → reduce N

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
- **Expected:** 20-30% TTFT reduction with grammar

**Experiment 3: Microbenchmarks**

- **Setup:** Synthetic workloads
- **Metrics:** Tag extraction speed, boundary detection, memory lookup
- **Expected:** ≥10× improvement on at least one

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

**Ablation 3: Grammar Impact**

- With grammar constraints
- Without (post-hoc validation)

**Ablation 4: Model Sensitivity** (if time)

- Qwen2.5-14B (primary)
- Llama-3.1-8B
- Mistral-7B

---

## 8) Paper Positioning

### Title Options

1. "TerseTalk: Grammar-Constrained JSONL Protocols for Efficient Multi-Agent Communication"
2. "Making Structure Faster: How Typed Protocols and Grammar Constraints Accelerate Multi-Agent LLMs"
3. "From Free-form to Fast: A Systems Approach to Multi-Agent Communication"

### Abstract Structure (150 words)

```
Multi-agent LLM systems waste tokens on verbose free-form messages.
We present TerseTalk-JSONL, a typed protocol with grammar constraints.
Key insight: structured generation can be faster, not slower.
Contributions: (1) JSONL protocol with overflow pointers,
(2) grammar constraints reducing TTFT by 20-30%,
(3) 10× faster tag extraction vs free-form.
Evaluation on HotpotQA/GSM8K shows 25-35% token reduction
with <2% quality loss. Open-source implementation included.
```

### Key Claims

1. **"Structured generation makes multi-agent communication faster"**
2. **"Typed protocols enable 10× faster message processing"**
3. **"Grammar constraints reduce latency while ensuring compliance"**

---

## 9) Appendices

### A. Glossary

**Terms:**

- **JSONL/NDJSON:** One JSON value per line, streaming-friendly format
- **M# references:** Memory pointers for overflowed content (like footnotes)
- **Soft caps:** Target lengths triggering overflow, not hard truncation
- **Grammar constraints:** Force model to generate valid syntax only
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
    --grammar

# Run full evaluation
python scripts/run_v05.py \
    --task hotpotqa \
    --system tersetalk \
    --n 500 \
    --seed 42 \
    --caps '{"f":30,"p":20,"q":30}' \
    --model qwen2.5-14b-instruct-q4_k_m.gguf

# Generate figures
python scripts/analyze_v05.py \
    --indir results \
    --outdir figures

# Run microbenchmarks
python benchmarks/run_all.py

# Clean old results
python scripts/cleanup.py --keep-last 5
```

### C. Model Setup Guide

```python
# Download Qwen2.5-14B GGUF
wget https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf

# Test grammar constraints
from tersetalk.constrained_generator import ConstrainedJSONLGenerator

gen = ConstrainedJSONLGenerator("qwen2.5-14b-instruct-q4_k_m.gguf")
result = gen.generate("Generate a fact in JSONL:", max_tokens=50)
print(result)  # Should be valid ["f", "..."]
```

---

## Timeline Summary

**Week 1-2:** Protocol + Grammar (MUST work)
**Week 3-4:** Microbenchmarks (MUST show 10× somewhere)
**Week 5-6:** Full evaluation (MUST show 25% reduction)
**Week 7:** Topology (ONLY if ahead)
**Week 8:** Writing (focus on systems narrative)

**Go/No-Go Decisions:**

- End of Week 1: Grammar working? If no → pivot to post-hoc only
- End of Week 3: Microbenchmarks promising? If no → adjust claims
- End of Week 5: On track? If no → skip topology entirely

**Success Criteria:**

- Grammar constraints reduce TTFT by ≥20%
- At least one 10× microbenchmark win
- 25-35% token reduction with <2% quality loss
- Reproducible with fixed seeds
- Clean, documented code with tests

---

## Final Checklist Before Starting

- [ ] Download Qwen2.5-14B-GGUF model
- [ ] Test llama-cpp-python installation
- [ ] Verify grammar constraints work on simple example
- [ ] Set up results directory structure
- [ ] Initialize git repo with .gitignore
- [ ] Create requirements.txt
- [ ] Write initial README
- [ ] Set up pytest with first dummy test
- [ ] Configure VSCode/IDE with Python environment
- [ ] Block calendar for focused coding time

**Start Date:** ****\_\_\_****
**Target Submission:** MLSys 2026 (2 months)

---

_This proposal represents a systems-first approach to multi-agent communication, emphasizing measurable performance improvements and practical implementation over theoretical contributions._

# 16) Extension (v1.0): Binary Task‑Adaptive Topology (scoped)

> Keep v0.5 unchanged as the core result. v1.0 adds a **minimal online topology switcher** that leverages TerseTalk‑JSONL signals. We position this as **online, instance‑adaptive** (per example), contrasting with **offline topology design** like **G‑Designer**.

## v1.0 goals (narrow scope)

- **Topologies:** support **Direct Mode (Star)** and **Collaborative Mode (Chain+Verify)**.
- **Router:** simple **binary decision** using typed signals (counts of `f/u/p/q`, presence of `d:M#` and code fences).
- **Metrics:** topology selection accuracy (vs task labels), token/latency, overflow rate, quality.
- **Safety valves:** cap switch frequency (min‑dwell 1 round), cap fan‑out.

### Benchmarks suited for topology differences

We evaluate inside **MARBLE (MultiAgentBench)** and staple tasks where the **optimal topology** is intuitive:  
**Direct‑optimal**: **BoolQ**, **CommonsenseQA**, **SocialIQA**.  
**Collaborative‑optimal**: **HotpotQA**, **GSM8K**, **HumanEval**.  
(Optionally add SQuAD for reading‑comprehension variety.)

### Core design: two topologies + one explainable router

**Topologies**

- **Direct Mode (Star):** Manager → Worker → Critic (good for single‑path Q/A)
- **Collaborative Mode (Chain+Verify):** Manager → Worker₁ → Worker₂ → Critic → Manager → Worker₁ (one verify/fix loop; good for multi‑step/code)

**Router using TerseTalk‑JSONL signals** (≈200 LOC in `PR‑V1.1`)

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

- Fixed‑Star (always Star)
- Fixed‑Chain (always Chain+Verify)
- Random‑Switch (50/50)
- Oracle (task‑label topology; upper bound)
- FreeForm‑Adaptive (same binary rules on free‑form messages)
- **TerseTalk‑Adaptive (ours)** (binary rules on typed protocol)

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
- Protocol advantage: ~**2×** better selection accuracy vs free‑form (typed fields expose `p/f/q/d` cleanly).

### Implementation plan (2 weeks additional)

**Week v1.1 — Router + collaborative executor**

- **PR‑V1.1: `topology_router.py` (~200 LOC)** — binary router (above); tests: synthetic JSONL → topology; threshold monotonicity.
- **Add `run_collaborative_once(...)` (~150 LOC)** — single verify/fix loop; EchoModel test; metrics match `run_star_once` (+signals).

**Week v1.2 — MARBLE runner**

- **PR‑V1.2: `scripts/run_marble_v1.py` (~250 LOC)** — run all 6 systems across 6 tasks, N=100 each; outputs per‑system CSVs + summary JSON.
- **Ablations:** θ ∈ {1,3,5,7}; ablate `{facts,plans,questions,has_refs,has_code}`; protocol necessity (scrambled tags); leave‑one‑task‑out.
- **Failure‑mode checks:** phase/session tags enforce ordering; deref depth/τ guard; min‑dwell=1 round to avoid thrash.

---

## 17) Optional v1.1: Density‑triggered router & sparse mesh (if time allows)

- **Density‑Adaptive Router** — use protocol **information density** slope to choose between star/chain/mesh.
- **Adaptive Mesh (sparse)** — prune a complete graph using **protocol affinity** (e.g., similar tag patterns), ensure connectivity; single round only.
- **Metrics:** add **routing decision ms**, **switch latency**, **avg path length**.

> These are compelling but **out of scope** for the initial v1.0 binary target; include only if schedule permits.

---

## 18) Deployment (optional, nice‑to‑have for artifact badge)

Provide a simple container setup so others can run echo‑based tests and CPU microbenchmarks.

**PR‑D‑01 — docker-compose (≤120 LOC)**

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

**PR‑D‑02 — Makefile helpers (≤80 LOC)**  
Targets: `install`, `test`, `run-synth`, `bench`.

---

## Appendix A — Glossary (with plain‑English tie‑ins)

- **JSONL / NDJSON.** “One JSON value per line”; streaming‑friendly and easy to validate. (Each line is an independent record.)
- **Soft cap.** A target length per field; try to compress; if still too long, emit an `["o", summary, "M#id"]` overflow line. (Don’t truncate meaning.)
- **M# memory.** Bounded key‑value store for overflowed text; entries referenced by IDs like `M#23`. (Like footnotes.)
- **Semantic preservation (SP).** How much meaning the typed message retains vs. the free‑form baseline (BERTScore/BLEURT). (Does the compressed message still say the same thing?)
- **Star / Chain / Flat.** Canonical agent topologies for message passing. (Star = hub‑and‑spoke; Chain = pipeline; Flat = everyone talks in one round.)
- **Topology switching.** Choosing a topology per instance, online. (Flip between Star/Chain/Flat based on simple signals.)
- **Prompt compression.** Techniques like LLMLingua/‑2 that shorten prompts while trying to preserve meaning.
- **Structured decoding.** Enforcing output shapes (JSON schema/CFG) during generation (e.g., vLLM or llama.cpp grammars). (v0.5 is post‑hoc; v1.x can enable at generation.)
- **TPS / TTFT / ITL.** Throughput (tokens/s), Time‑to‑First‑Token, and Inter‑Token Latency (per‑token pace after TTFT).

---

## One‑slide TL;DR (for your PI)

- **v0.5 = protocol‑only:** TerseTalk‑JSONL (typed, compact, soft‑capped, overflow pointers).
- **Baselines:** free‑form and free‑form + LLMLingua/‑2.
- **Tight PRs (≤250 LOC)**, GPU‑free CI via `EchoModel`.
- **Metrics:** tokens, EM/Acc, SP (BERTScore), overflow/memory, JSONL serde latency (optional TPS/TTFT/ITL).
- **v1.0 extension:** binary topology switching (Direct vs Collaborative) guided by TerseTalk signals, evaluated in MARBLE; selection/efficiency metrics; compares vs fixed and random baselines.
- **Optionals that strengthen MLSys submission:** grammar‑constrained JSONL, streaming processing, microbenchmarks, industrial protocol baselines, and density‑aware routing.
- **Positioning:** protocol–topology **co‑design**; complements offline topology design (G‑Designer) and message compression (LLMLingua).

---

## (Optional) Queueing theory context

If capacity allows, add a light queueing‑style utilization backdrop (M/M/c) to reason about expected wait vs. fan‑out across Star/Chain. Treat as qualitative context only; do **not** overclaim.
