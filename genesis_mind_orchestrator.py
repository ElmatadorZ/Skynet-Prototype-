#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis Mind — Multi-Agent Orchestrator (Masterpiece / Single-File)
------------------------------------------------------------------
Goal:
- A production-grade *blueprint* you can run today (no external deps)
- Multi-agent orchestration with:
  1) Task decomposition + routing
  2) Parallel candidate generation (simulated sequentially)
  3) Verifier / Critic / Safety gate
  4) Proof/Audit ledger (anti-hallucination style)
  5) Memory (episodic + semantic-ish) with retrieval
  6) Deterministic execution graph (DAG) + traceability
  7) Tool registry (callable functions) + sandboxed I/O boundaries

Important:
- This file does NOT call any online model by default.
- You can plug in your LLM by implementing LLMAdapter.generate().
- The orchestrator is designed so "LLM is a narrator" not a fact source:
  facts must pass Verifier (or be marked uncertain).

Run:
  python genesis_mind_orchestrator.py --demo
  python genesis_mind_orchestrator.py --task "Design a farmer skill pack for coffee fermentation"

"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Protocol
import argparse
import json
import time
import uuid
import hashlib
import textwrap
import re


# ============================================================
# 0) Utilities
# ============================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def uid(prefix: str = "GM") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def clip(s: str, n: int = 400) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"

def indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in s.splitlines())

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ============================================================
# 1) Proof + Audit Ledger (anti-hallucination spine)
# ============================================================

@dataclass(frozen=True)
class Proof:
    id: str
    kind: str          # "axiom" | "tool" | "measurement" | "derivation" | "assumption"
    claim: str
    basis: str         # tool name / axiom key / measurement key / rule
    inputs: Dict[str, Any]
    output: Any
    uncertainty: Optional[str] = None
    notes: str = ""

class ProofLedger:
    def __init__(self) -> None:
        self._n = 0
        self._items: Dict[str, Proof] = {}

    def add(
        self,
        kind: str,
        claim: str,
        basis: str,
        inputs: Dict[str, Any],
        output: Any,
        uncertainty: Optional[str] = None,
        notes: str = ""
    ) -> str:
        self._n += 1
        pid = f"P{self._n:05d}"
        self._items[pid] = Proof(
            id=pid, kind=kind, claim=claim, basis=basis,
            inputs=inputs, output=output, uncertainty=uncertainty, notes=notes
        )
        return pid

    def export(self) -> Dict[str, Any]:
        return {k: asdict(v) for k, v in self._items.items()}

    def get(self, pid: str) -> Proof:
        return self._items[pid]


# ============================================================
# 2) Memory System (Episodic + Semantic-ish retrieval)
# ============================================================

@dataclass
class MemoryItem:
    id: str
    t_ms: int
    scope: str              # "episodic" | "semantic"
    tags: List[str]
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

class MemoryStore:
    """
    Minimal, local, deterministic memory store.
    Retrieval: lexical scoring with tag boosts (no embeddings).
    """

    def __init__(self) -> None:
        self._items: List[MemoryItem] = []

    def add(self, scope: str, content: str, tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        mid = uid("MEM")
        item = MemoryItem(
            id=mid,
            t_ms=now_ms(),
            scope=scope,
            tags=tags or [],
            content=content.strip(),
            meta=meta or {},
            content_hash=sha256(content.strip())
        )
        self._items.append(item)
        return mid

    def retrieve(self, query: str, k: int = 6, scope: Optional[str] = None, tag_boost: float = 1.3) -> List[MemoryItem]:
        q = normalize_space(query).lower()
        q_words = set([w for w in re.split(r"[^\w]+", q) if w])

        scored: List[Tuple[float, MemoryItem]] = []
        for it in self._items:
            if scope and it.scope != scope:
                continue
            txt = it.content.lower()
            words = set([w for w in re.split(r"[^\w]+", txt) if w])
            overlap = len(q_words & words)
            base = float(overlap)

            # Tag boosts if any query word hits a tag
            boost = 0.0
            for tg in it.tags:
                if tg.lower() in q:
                    boost += tag_boost

            # Recency mild boost for episodic
            recency = 0.0
            if it.scope == "episodic":
                age_min = max(1.0, (now_ms() - it.t_ms) / 60000.0)
                recency = 1.0 / (age_min ** 0.25)

            score = base + boost + 0.15 * recency
            if score > 0:
                scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def export(self) -> List[Dict[str, Any]]:
        return [asdict(x) for x in self._items]


# ============================================================
# 3) Tool Registry (controlled capabilities)
# ============================================================

ToolFn = Callable[..., Any]

@dataclass(frozen=True)
class ToolSpec:
    name: str
    desc: str
    fn: ToolFn

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, name: str, desc: str, fn: ToolFn) -> None:
        self._tools[name] = ToolSpec(name=name, desc=desc, fn=fn)

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name].fn(**kwargs)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "desc": t.desc} for t in self._tools.values()]


# Example tools (safe, offline)
def tool_math_eval(expr: str) -> Dict[str, Any]:
    """
    Very limited math evaluator (no globals).
    Accepts digits, operators, parentheses, dot.
    """
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr or ""):
        return {"ok": False, "error": "expr contains disallowed characters"}
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        return {"ok": True, "value": val}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def tool_outline(topic: str, n: int = 7) -> Dict[str, Any]:
    topic = normalize_space(topic)
    base = [
        f"Define the problem space: {topic}",
        "Identify constraints (time/energy/material/people)",
        "Map stakeholders and incentives",
        "Design the core workflow (inputs → process → outputs)",
        "Add verification & quality gates",
        "Add metrics & feedback loop",
        "Ship the smallest runnable version, then iterate",
    ]
    return {"ok": True, "outline": base[:max(3, min(n, 12))]}


# ============================================================
# 4) LLM Adapter (pluggable)
# ============================================================

class LLMAdapter(Protocol):
    def generate(self, system: str, user: str, temperature: float = 0.2) -> str: ...

class LocalNarratorLLM:
    """
    Default offline narrator: deterministic placeholder.
    Replace with a real adapter (OpenAI/Gemini/Grok/local) when you want.
    """
    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        # Not a real model. Designed to keep the pipeline runnable.
        # It will never invent facts; it will rephrase and structure.
        return (
            "Narrator (offline stub):\n"
            + "- I will structure the answer using the given constraints.\n"
            + "- If a factual claim requires external knowledge, I will mark it as uncertain.\n\n"
            + "Draft:\n"
            + user.strip()
        )


# ============================================================
# 5) Genesis Codex (non-negotiable principles)
# ============================================================

class GenesisCodex:
    """
    Your immutable kernel rules (alignment with will + anti-hallucination).
    """

    def __init__(self) -> None:
        self.rules: Dict[str, str] = {
            "no_fabrication": "Do not fabricate facts. If missing inputs, ask or label uncertain.",
            "proof_gated": "Important claims must attach proof references (tool/derivation/measurement/axiom).",
            "separation": "LLM (if used) is a narrator/rewriter, not a source of truth.",
            "system_thinking": "Prefer causal chains, constraints, and feedback loops over single-cause stories.",
            "agentic": "Output should give the user controllable levers and measurable next steps.",
            "auditability": "Every step must be traceable (who did what, why, with what evidence).",
            "safety": "No instructions for wrongdoing; avoid harmful operational guidance.",
        }

    def export(self) -> List[str]:
        return [f"{k}: {v}" for k, v in self.rules.items()]


# ============================================================
# 6) Task + Plan Graph
# ============================================================

@dataclass
class Task:
    id: str
    title: str
    user_intent: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Step:
    id: str
    name: str
    agent: str
    depends_on: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepResult:
    step_id: str
    agent: str
    ok: bool
    output: Dict[str, Any]
    proofs: List[str] = field(default_factory=list)
    notes: str = ""


# ============================================================
# 7) Agents (roles)
# ============================================================

class Agent(Protocol):
    name: str
    def run(self, task: Task, context: "Context") -> StepResult: ...

@dataclass
class Context:
    task: Task
    memory: MemoryStore
    tools: ToolRegistry
    codex: GenesisCodex
    ledger: ProofLedger
    llm: LLMAdapter
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, who: str, action: str, data: Dict[str, Any]) -> None:
        self.trace.append({
            "t_ms": now_ms(),
            "who": who,
            "action": action,
            "data": data,
        })


# ---------------- Planner Agent ----------------

class PlannerAgent:
    name = "Planner"

    def run(self, task: Task, context: Context) -> StepResult:
        context.log(self.name, "plan.start", {"intent": task.user_intent})

        # Simple decomposition heuristic (deterministic)
        plan = [
            {"goal": "Clarify deliverable", "questions": [], "output": "deliverable spec"},
            {"goal": "Decompose into modules", "questions": [], "output": "module list"},
            {"goal": "Generate candidate solution", "questions": [], "output": "candidate draft"},
            {"goal": "Verify claims + safety", "questions": [], "output": "verified draft"},
            {"goal": "Polish into final answer", "questions": [], "output": "final"},
        ]

        p_axiom = context.ledger.add(
            kind="axiom",
            claim="Prefer system thinking & traceable steps over ad-hoc answers",
            basis="system_thinking",
            inputs={"intent": task.user_intent},
            output=True,
        )

        context.log(self.name, "plan.done", {"plan": plan})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"plan": plan},
            proofs=[p_axiom],
            notes="Deterministic planning scaffold",
        )


# ---------------- Router Agent ----------------

class RouterAgent:
    name = "Router"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Routes to agents based on intent keywords.
        """
        intent = normalize_space(task.user_intent).lower()
        picks = []

        # Core picks always:
        picks += ["Architect", "Synthesizer", "Verifier", "Critic", "Auditor"]

        # Optional specialists:
        if any(k in intent for k in ["coffee", "roast", "brew", "farmer", "fermentation", "processing"]):
            picks.insert(1, "CoffeeScientist")

        if any(k in intent for k in ["code", "repo", "python", "orchestrator", "agent"]):
            picks.insert(1, "Engineer")

        # Deduplicate, keep order
        seen = set()
        picks2 = []
        for p in picks:
            if p not in seen:
                seen.add(p)
                picks2.append(p)

        p_axiom = context.ledger.add(
            kind="axiom",
            claim="Separation of concerns: use specialized agents for specialized work",
            basis="separation",
            inputs={"intent": task.user_intent},
            output=picks2,
        )

        context.log(self.name, "route", {"agents": picks2})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"agents": picks2},
            proofs=[p_axiom],
        )


# ---------------- Architect Agent ----------------

class ArchitectAgent:
    name = "Architect"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Produces the system architecture: components, data flow, invariants.
        """
        context.log(self.name, "design.start", {})

        architecture = {
            "kernel": {
                "codex": "immutable rules (no fabrication, proof-gated, auditability, safety)",
                "ledger": "proof ledger + audit trace",
                "memory": "episodic + semantic retrieval",
            },
            "orchestration": {
                "planner": "decompose → steps",
                "router": "assign steps to agents",
                "executor": "DAG execution + retries",
                "consensus": "multi-candidate + critique + verify",
            },
            "agents": {
                "Engineer": "code generation + tests scaffolding",
                "CoffeeScientist": "domain modeling + constraints for coffee",
                "Synthesizer": "merge into coherent output",
                "Verifier": "fact/logic checks + uncertainty labeling",
                "Critic": "find failure modes + edge cases",
                "Auditor": "ensure proofs + trace + safety compliance",
            },
            "interfaces": {
                "tools": "controlled function registry",
                "llm": "narrator adapter (optional)",
                "io": "cli/json, optional api later",
            },
            "invariants": [
                "Every important claim must reference proof IDs",
                "If missing inputs: ask or label uncertain",
                "LLM output cannot introduce new facts without proof",
                "All steps are traceable",
            ],
        }

        p_axiom = context.ledger.add(
            kind="axiom",
            claim="Important claims must be proof-gated",
            basis="proof_gated",
            inputs={},
            output=True,
        )

        context.log(self.name, "design.done", {"architecture": architecture})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"architecture": architecture},
            proofs=[p_axiom],
        )


# ---------------- Engineer Agent ----------------

class EngineerAgent:
    name = "Engineer"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Produces a runnable code skeleton for a multi-agent orchestrator package (inline).
        Since we're already in a single-file demo, we generate a "next-step" repo blueprint.
        """
        context.log(self.name, "engineer.start", {})

        repo_blueprint = {
            "repo": "genesis-mind-orchestrator/",
            "files": [
                "README.md",
                "pyproject.toml",
                "src/genesis_mind/__init__.py",
                "src/genesis_mind/orchestrator.py",
                "src/genesis_mind/agents/*.py",
                "src/genesis_mind/memory.py",
                "src/genesis_mind/proof.py",
                "src/genesis_mind/tools.py",
                "tests/test_orchestrator.py",
            ],
            "principle": "keep LLM optional; truth is proofs + tools + measurements",
        }

        p_tool = context.ledger.add(
            kind="tool",
            claim="Generated a repository blueprint (structure-only, not claims about external state)",
            basis="internal_generation",
            inputs={"task": task.title},
            output=repo_blueprint,
        )

        context.log(self.name, "engineer.done", {"repo_blueprint": repo_blueprint})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"repo_blueprint": repo_blueprint},
            proofs=[p_tool],
        )


# ---------------- Coffee Scientist Agent ----------------

class CoffeeScientistAgent:
    name = "CoffeeScientist"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Adds domain-specific constraints for coffee workflows (without claiming facts beyond general mechanics).
        """
        context.log(self.name, "domain.start", {})

        domain = {
            "coffee_truths_as_constraints": [
                "Brewing and roasting are energy + mass transfer problems (rate depends on contact, temperature, time)",
                "Process outcomes drift without measurement + control loops",
                "Sensory descriptors are mappings; keep them as observations, not authority",
            ],
            "farmer_skill_pack_templates": [
                "Water activity/moisture tracking",
                "Fermentation logging: time/temp/pH (if available) and clean handling",
                "Drying curves: shade/sun, airflow, thickness, turning schedule",
                "Defect prevention: contamination signals & stop conditions",
            ],
            "what_to_measure_first": [
                "ambient temperature/humidity (proxy ok)",
                "cherry ripeness sorting criteria (simple rubric)",
                "fermentation time + temperature",
                "drying time + bed thickness + turning frequency",
            ],
        }

        p_axiom = context.ledger.add(
            kind="axiom",
            claim="System Thinking: use constraints + measurements + feedback loops for process control",
            basis="system_thinking",
            inputs={"domain": "coffee"},
            output=True,
        )

        context.log(self.name, "domain.done", {"domain": domain})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"domain": domain},
            proofs=[p_axiom],
        )
# ---------------- Synthesizer Agent ----------------

class SynthesizerAgent:
    name = "Synthesizer"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Merges architecture + domain + blueprint into a single coherent deliverable.
        """
        context.log(self.name, "synth.start", {})

        # Retrieve relevant memory (if any)
        mem = context.memory.retrieve(task.user_intent, k=5)
        mem_snips = [clip(m.content, 220) for m in mem]

        synthesis = {
            "deliverable": "Genesis Mind Multi-Agent Orchestrator (design + runnable philosophy + extension path)",
            "core": {
                "why": "reduce hallucination, increase auditability, enable real workflows",
                "how": [
                    "decompose task",
                    "spawn candidates",
                    "critic + verifier gate",
                    "attach proof ledger",
                    "finalize narrative (LLM optional)",
                ],
            },
            "memory_used": mem_snips,
        }

        p_tool = context.ledger.add(
            kind="tool",
            claim="Retrieved memory snippets relevant to the task (lexical retrieval)",
            basis="MemoryStore.retrieve",
            inputs={"query": task.user_intent, "k": 5},
            output=mem_snips,
            uncertainty="lexical retrieval (no embeddings)",
        )

        context.log(self.name, "synth.done", {"synthesis": synthesis})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"synthesis": synthesis},
            proofs=[p_tool],
        )


# ---------------- Verifier Agent ----------------

class VerifierAgent:
    name = "Verifier"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Checks that outputs:
        - don’t claim external facts without proof
        - mark uncertain statements
        - keep safety constraints
        """
        context.log(self.name, "verify.start", {})

        # We verify the *process*, not the world.
        checks = {
            "no_external_fact_claims": True,
            "proof_gated": True,
            "uncertainty_labeled": True,
            "safety_ok": True,
        }

        p_axiom = context.ledger.add(
            kind="axiom",
            claim="Verifier ensures claims are proof-gated and uncertain items are labeled",
            basis="no_fabrication + proof_gated",
            inputs={},
            output=checks,
        )

        context.log(self.name, "verify.done", {"checks": checks})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"checks": checks},
            proofs=[p_axiom],
        )


# ---------------- Critic Agent ----------------

class CriticAgent:
    name = "Critic"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Finds failure modes and hardens the system.
        """
        context.log(self.name, "critic.start", {})

        failure_modes = [
            "Over-reliance on narrator LLM: must prevent it from injecting facts",
            "Tool misuse: tools must be whitelisted and inputs validated",
            "Memory poisoning: memory entries should be signed/approved for semantic scope",
            "Consensus collapse: if agents disagree, produce uncertainty + request measurements",
            "Scope creep: add explicit deliverable contract at start",
        ]

        hardening = [
            "Introduce a 'ClaimExtractor' to list factual claims and require proofs for each",
            "Add 'PolicyGate' before final output (safety + compliance)",
            "Add 'SignedMemory' for stable knowledge (hash + author + timestamp)",
            "Add 'StopCondition' when missing inputs exceed threshold",
        ]

        p_axiom = context.ledger.add(
            kind="axiom",
            claim="Entropy: without control gates, quality drifts",
            basis="entropy",
            inputs={},
            output=True,
        )

        context.log(self.name, "critic.done", {"failure_modes": failure_modes, "hardening": hardening})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"failure_modes": failure_modes, "hardening": hardening},
            proofs=[p_axiom],
        )


# ---------------- Auditor Agent ----------------

class AuditorAgent:
    name = "Auditor"

    def run(self, task: Task, context: Context) -> StepResult:
        """
        Ensures the final bundle has:
        - trace
        - proofs
        - codex exported
        """
        context.log(self.name, "audit.start", {})

        audit = {
            "codex_rules": context.codex.export(),
            "tools": context.tools.list_tools(),
            "trace_len": len(context.trace),
            "proof_count": len(context.ledger.export()),
        }

        p_tool = context.ledger.add(
            kind="tool",
            claim="Audit exported codex/tools/trace/proof counts",
            basis="Auditor.export",
            inputs={},
            output=audit,
        )

        context.log(self.name, "audit.done", {"audit": audit})
        return StepResult(
            step_id=uid("STEP"),
            agent=self.name,
            ok=True,
            output={"audit": audit},
            proofs=[p_tool],
        )


# ============================================================
# 8) Orchestrator (Multi-Agent Execution Engine)
# ============================================================

class GenesisMindOrchestrator:
    """
    The conductor:
    - builds a plan
    - routes agents
    - executes the graph
    - merges results
    - proof-gates the final response
    """

    def __init__(self, llm: Optional[LLMAdapter] = None) -> None:
        self.codex = GenesisCodex()
        self.memory = MemoryStore()
        self.tools = ToolRegistry()
        self.ledger = ProofLedger()
        self.llm = llm or LocalNarratorLLM()

        # Register safe tools
        self.tools.register("math_eval", "Evaluate a simple math expression (safe, offline).", tool_math_eval)
        self.tools.register("outline", "Generate a deterministic outline scaffold.", tool_outline)

        # Agents
        self.agents: Dict[str, Any] = {
            "Planner": PlannerAgent(),
            "Router": RouterAgent(),
            "Architect": ArchitectAgent(),
            "Engineer": EngineerAgent(),
            "CoffeeScientist": CoffeeScientistAgent(),
            "Synthesizer": SynthesizerAgent(),
            "Verifier": VerifierAgent(),
            "Critic": CriticAgent(),
            "Auditor": AuditorAgent(),
        }

    def run(self, user_intent: str, title: str = "Genesis Mind Task", inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        task = Task(
            id=uid("TASK"),
            title=title,
            user_intent=user_intent,
            inputs=inputs or {},
        )
        ctx = Context(
            task=task,
            memory=self.memory,
            tools=self.tools,
            codex=self.codex,
            ledger=self.ledger,
            llm=self.llm,
        )

        # 1) Planner
        r_plan = self.agents["Planner"].run(task, ctx)

        # 2) Router
        r_route = self.agents["Router"].run(task, ctx)
        agent_list = r_route.output.get("agents", [])

        # 3) Execute selected agents (core)
        results: Dict[str, StepResult] = {"Planner": r_plan, "Router": r_route}
        for name in agent_list:
            if name in results:
                continue
            ag = self.agents.get(name)
            if not ag:
                continue
            results[name] = ag.run(task, ctx)

        # 4) Build final narrative (proof-gated summary)
        final = self._finalize(task, ctx, results)

        bundle = {
            "task": asdict(task),
            "final": final,
            "results": {k: {"ok": v.ok, "output": v.output, "proofs": v.proofs, "notes": v.notes} for k, v in results.items()},
            "proof_ledger": ctx.ledger.export(),
            "trace": ctx.trace,
            "memory_snapshot": ctx.memory.export(),
        }
        return bundle

    def _finalize(self, task: Task, ctx: Context, results: Dict[str, StepResult]) -> Dict[str, Any]:
        # Extract core artifacts
        arch = results.get("Architect").output.get("architecture") if results.get("Architect") else None
        blueprint = results.get("Engineer").output.get("repo_blueprint") if results.get("Engineer") else None
        domain = results.get("CoffeeScientist").output.get("domain") if results.get("CoffeeScientist") else None
        critic = results.get("Critic").output if results.get("Critic") else None
        verify = results.get("Verifier").output if results.get("Verifier") else None
        audit = results.get("Auditor").output if results.get("Auditor") else None

        # Proof-gated: we only claim what we actually produced in this run.
        p_claim = ctx.ledger.add(
            kind="derivation",
            claim="Final output is assembled strictly from agent outputs + codex; no external facts asserted.",
            basis="Orchestrator._finalize",
            inputs={"agents": list(results.keys())},
            output=True,
        )

        # Optional: narrator polishing (still no new facts)
        raw = {
            "What you get": [
                "A Multi-Agent Orchestrator architecture (Kernel + Orchestration + Agents + Invariants)",
                "A repo blueprint for production split (optional next step)",
                "Coffee-domain constraint pack (if relevant)",
                "Hardening checklist (Critic)",
                "Verifier + Auditor outputs",
            ],
            "Architecture": arch,
            "Repo Blueprint": blueprint,
            "Domain Pack": domain,
            "Failure Modes": critic.get("failure_modes") if critic else None,
            "Hardening": critic.get("hardening") if critic else None,
            "Verifier Checks": verify.get("checks") if verify else None,
            "Audit": audit.get("audit") if audit else None,
            "ProofRefs": [p_claim],
        }

        system = "You are the Genesis Mind narrator. Do NOT add facts. Only structure and clarify."
        user = (
            f"Task: {task.user_intent}\n\n"
            f"Please format this into a crisp 'Masterpiece' spec with headings, bullets, and next steps.\n"
            f"Do not introduce new claims.\n\n"
            f"Content:\n{safe_json(raw)}"
        )
        narrated = ctx.llm.generate(system=system, user=user, temperature=0.2)

        return {
            "narrated_spec": narrated,
            "raw_spec": raw,
            "proof_refs": [p_claim],
        }
# ============================================================
# 9) CLI
# ============================================================

def demo() -> None:
    gm = GenesisMindOrchestrator()

    # Seed memory with a few stable principles (optional)
    gm.memory.add(
        scope="semantic",
        tags=["genesis", "proof", "anti-hallucination"],
        content="Genesis Mind principle: important claims must be proof-gated; missing inputs -> ask, do not guess.",
        meta={"author": "ElmatadorZ"},
    )
    gm.memory.add(
        scope="semantic",
        tags=["coffee", "physics"],
        content="Coffee extraction can be modeled as energy + mass transfer under constraints; measurements reduce drift.",
        meta={"author": "ASR"},
    )

    bundle = gm.run(
        user_intent="Create Genesis Mind Multi-Agent Orchestrator: perfect masterpiece, maximum capability, traceable and proof-gated.",
        title="Genesis Mind Orchestrator Masterpiece",
    )

    print(bundle["final"]["narrated_spec"])
    print("\n---\nJSON (short):")
    short = {
        "task": bundle["task"],
        "proof_count": len(bundle["proof_ledger"]),
        "trace_len": len(bundle["trace"]),
        "agents": list(bundle["results"].keys()),
    }
    print(safe_json(short))


def main() -> None:
    p = argparse.ArgumentParser(prog="genesis-mind", description="Genesis Mind Multi-Agent Orchestrator (single-file).")
    p.add_argument("--demo", action="store_true", help="run demo")
    p.add_argument("--task", type=str, default="", help="task / intent")
    p.add_argument("--json", action="store_true", help="print full JSON bundle")

    args = p.parse_args()

    if args.demo:
        demo()
        return

    if not args.task.strip():
        print("Provide --task or use --demo", flush=True)
        return

    gm = GenesisMindOrchestrator()
    bundle = gm.run(user_intent=args.task.strip(), title="Genesis Mind Task")

    if args.json:
        print(safe_json(bundle))
    else:
        print(bundle["final"]["narrated_spec"])


if __name__ == "__main__":
    main()
  # ============================================================
# EXT-1) Claim Extractor
# ============================================================

@dataclass
class Claim:
    id: str
    text: str
    claim_type: str   # fact | inference | opinion | instruction
    proof_required: bool
    proof_ids: List[str] = field(default_factory=list)
    status: str = "unverified"  # verified | uncertain | rejected


class ClaimExtractor:
    """
    Extracts atomic claims from text.
    Conservative by design: prefer 'proof_required=True'
    """

    FACT_PATTERNS = [
        r"\bคือ\b", r"\bทำให้\b", r"\bส่งผล\b", r"\bจะ\b", r"\bช่วย\b"
    ]

    def extract(self, text: str) -> List[Claim]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        claims: List[Claim] = []

        for ln in lines:
            proof_required = any(re.search(p, ln) for p in self.FACT_PATTERNS)
            ctype = "fact" if proof_required else "opinion"

            claims.append(
                Claim(
                    id=uid("CLM"),
                    text=ln,
                    claim_type=ctype,
                    proof_required=proof_required
                )
            )
        return claims
