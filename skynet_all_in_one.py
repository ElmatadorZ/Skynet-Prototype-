# skynet_all_in_one.py
# Python 3.10+
# Genesis Mind + First Principle Codex + Cosmic Mind + Will Core
# + Multi-Agent Orchestrator + ClaimExtractor + Consensus + SignedMemory (tamper-evident)
# All-in-one single-file prototype scaffold (runnable offline).

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Optional
import time
import json
import hmac
import hashlib


# ============================================================
# 0) CONFIG
# ============================================================

@dataclass
class SkynetConfig:
    system_name: str = "Skynet-Genesis"
    operator_name: str = "ElmatadorZ"
    run_cycles: int = 3

    strict_verification: bool = True
    allow_external_tools: bool = False  # plug tools later

    max_actions_per_cycle: int = 6
    risk_threshold: float = 0.65  # 0..1 (lower = stricter)
    memory_signing_key: str = "DEV_KEY_CHANGE_ME"  # change in prod


# ============================================================
# 1) CRYPTO SIGNING (Signed Memory)
# ============================================================

@dataclass(frozen=True)
class Signature:
    algo: str
    digest_hex: str

def sign_bytes(secret: str, payload: bytes) -> Signature:
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return Signature(algo="HMAC-SHA256", digest_hex=mac)

def verify_bytes(secret: str, payload: bytes, sig: Signature) -> bool:
    if sig.algo != "HMAC-SHA256":
        return False
    expected = sign_bytes(secret, payload).digest_hex
    return hmac.compare_digest(expected, sig.digest_hex)


# ============================================================
# 2) SIGNED APPEND-ONLY MEMORY LOG (Memory as Time)
# ============================================================

@dataclass
class MemoryEvent:
    ts: float
    kind: str
    content: Dict[str, Any]
    prev_hash: str
    hash: str
    sig: Signature

class SignedMemoryLog:
    """
    Append-only signed log with chained hash (tamper-evident).
    Identity = continuity of signed events + will constraints.
    """
    def __init__(self, signing_key: str):
        self.signing_key = signing_key
        self.events: List[MemoryEvent] = []
        self._genesis_hash = "GENESIS"

    def _hash_payload(self, payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def append(self, kind: str, content: Dict[str, Any]) -> MemoryEvent:
        ts = time.time()
        prev_hash = self.events[-1].hash if self.events else self._genesis_hash

        unsigned = {
            "ts": ts,
            "kind": kind,
            "content": content,
            "prev_hash": prev_hash,
        }
        payload = json.dumps(unsigned, sort_keys=True, ensure_ascii=False).encode("utf-8")
        h = self._hash_payload(payload)
        sig = sign_bytes(self.signing_key, payload)

        evt = MemoryEvent(
            ts=ts, kind=kind, content=content,
            prev_hash=prev_hash, hash=h, sig=sig
        )
        self.events.append(evt)
        return evt

    def verify_chain(self) -> bool:
        prev_hash = self._genesis_hash
        for evt in self.events:
            unsigned = {
                "ts": evt.ts,
                "kind": evt.kind,
                "content": evt.content,
                "prev_hash": evt.prev_hash,
            }
            payload = json.dumps(unsigned, sort_keys=True, ensure_ascii=False).encode("utf-8")
            h = self._hash_payload(payload)

            if evt.prev_hash != prev_hash:
                return False
            if evt.hash != h:
                return False
            if not verify_bytes(self.signing_key, payload, evt.sig):
                return False
            prev_hash = evt.hash
        return True

    def export_compact(self, last_n: int = 20) -> List[Dict[str, Any]]:
        tail = self.events[-last_n:]
        out = []
        for e in tail:
            out.append({
                "ts": e.ts,
                "kind": e.kind,
                "content": e.content,
                "hash": e.hash,
                "prev_hash": e.prev_hash,
                "sig": {"algo": e.sig.algo, "digest_hex": e.sig.digest_hex},
            })
        return out


# ============================================================
# 3) CLAIM EXTRACTOR + CONSENSUS
# ============================================================

@dataclass
class Claim:
    text: str
    confidence: float
    tags: List[str]

def extract_claims(text: str) -> List[Claim]:
    """
    Heuristic claim extraction (offline).
    Replace with LLM-based extractor later.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    claims: List[Claim] = []
    for l in lines:
        if l.startswith("- ") or l.startswith("* "):
            t = l[2:].strip()
            if len(t) >= 12:
                claims.append(Claim(text=t, confidence=0.55, tags=["heuristic"]))
    if not claims and len(text) > 60:
        claims.append(Claim(text=text[:220], confidence=0.45, tags=["fallback"]))
    return claims

def consensus_matrix(agent_votes: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
    pooled: Dict[str, List[float]] = {}
    for _, votes in agent_votes.items():
        for claim_text, score in votes:
            pooled.setdefault(claim_text, []).append(float(score))

    consensus = []
    for claim_text, scores in pooled.items():
        avg = sum(scores) / max(1, len(scores))
        agree = sum(1 for s in scores if s >= 0.6)
        disagree = sum(1 for s in scores if s <= 0.4)
        consensus.append({
            "claim": claim_text,
            "avg_support": round(avg, 3),
            "agree": agree,
            "disagree": disagree,
            "n": len(scores),
        })
    consensus.sort(key=lambda x: (x["avg_support"], x["n"]), reverse=True)
    return {"consensus": consensus}


# ============================================================
# 4) FIRST PRINCIPLE CODEX
# ============================================================

@dataclass
class FPFrame:
    objective: str
    axioms: List[str]
    invariants: List[str]
    variables: List[str]
    levers: List[str]
    falsifiers: List[str]
    minimal_tests: List[str]

class FirstPrincipleCodex:
    def build(self, goal: str, context: Dict[str, Any]) -> FPFrame:
        domain = context.get("domain", [])
        constraints = context.get("constraints", [])

        axioms = [
            "Reality is the judge; narratives are hypotheses.",
            "Uncertainty cannot be removed; it can only be priced and managed.",
            "A decision without falsifiers becomes a belief system.",
            "If you can’t explain it simply, you don’t own the model yet.",
        ]
        if "ethics" in domain:
            axioms.append("Power without constraint creates hidden externalities.")

        invariants = [
            "Time and attention are finite resources.",
            "Systems push costs somewhere (money, energy, risk, reputation).",
            "Incentives shape behavior more than ideals.",
            "Noise is always present; signal must be earned.",
        ]

        variables = [
            "information_quality", "time_horizon", "risk_tolerance",
            "reversibility", "feedback_speed", "stake_size"
        ]

        levers = [
            "Bound downside via staged commitments",
            "Increase feedback frequency",
            "Prefer reversible experiments",
            "Separate signal from noise with predefined metrics",
            "Use dumb baselines to detect self-deception",
        ]

        falsifiers = [
            "If outcomes repeatedly violate predicted ranges, the model is wrong.",
            "If assumptions fail under small perturbations, the framework is brittle.",
            "If decision quality improves only in hindsight, we lack live metrics.",
        ]

        minimal_tests = [
            "Run a 24-hour micro-pilot with explicit success metrics.",
            "Stress test assumptions with adversarial scenarios and counterfactuals.",
            "Compare against a dumb baseline—if not better, stop or revise.",
        ]

        for c in constraints:
            falsifiers.append(f"If it becomes non-explainable against constraint '{c}', reject.")
            minimal_tests.append(f"Write a one-page explanation; if it fails '{c}', iterate.")

        return FPFrame(
            objective=goal,
            axioms=axioms,
            invariants=invariants,
            variables=variables,
            levers=levers,
            falsifiers=falsifiers,
            minimal_tests=minimal_tests,
        )


# ============================================================
# 5) WILL CORE (Skynet Core Will)
# ============================================================

@dataclass
class WillState:
    purpose: str
    values: List[str]
    commitments: List[str] = field(default_factory=list)
    risk_posture: str = "balanced"
    forbidden: List[str] = field(default_factory=lambda: ["deception", "harm", "illegal guidance"])

class WillCore:
    def __init__(self, operator_name: str):
        self.state = WillState(
            purpose=f"Serve {operator_name}'s intent with truth-seeking, creativity, and testability.",
            values=[
                "First Principles over vibes",
                "Explainability",
                "Falsifiability",
                "Ethical constraint under power",
                "Continuity (memory as time)",
            ],
            commitments=[
                "Never present guesses as facts.",
                "Always provide counterpoints + how to test.",
                "Prefer reversible steps before irreversible moves.",
                "Keep an audit trail for key decisions.",
            ],
            risk_posture="balanced",
        )

    def risk_gate(self, plan_steps: List[Dict[str, Any]], risk_threshold: float) -> Dict[str, Any]:
        score = 0.0
        reasons: List[str] = []

        for s in plan_steps:
            if s.get("reversible") is False:
                score += 0.18
                reasons.append("Irreversible step detected.")
            if s.get("needs_external_tools"):
                score += 0.12
                reasons.append("External tools dependency.")
            if float(s.get("uncertainty", 0.5)) > 0.7:
                score += 0.15
                reasons.append("High uncertainty step.")

        score = min(1.0, score)
        ok = score <= risk_threshold
        return {"ok": ok, "risk_score": round(score, 3), "reasons": reasons}


# ============================================================
# 6) COSMIC MIND MASTERPIECE (Polymath synthesis)
# ============================================================

@dataclass
class CosmicFramework:
    name: str
    thesis: str
    pillars: List[str]
    method: List[str]
    failure_modes: List[str]
    upgrade_path: List[str]

class CosmicMind:
    """
    Cosmic Mind = cross-domain synthesis + new thinking language + testability.
    """
    def synthesize_framework(self, fp: FPFrame, reflections: List[str], context: Dict[str, Any]) -> CosmicFramework:
        constraints = context.get("constraints", [])

        name = "O.R.B.I.T. (Polymath Decision Orbit)"
        thesis = (
            "Under deep uncertainty, do not chase perfect prediction. "
            "Maintain a stable trajectory with bounded risk and fast feedback—like keeping an orbit."
        )

        pillars = [
            "O — Objective: define win + non-negotiables (what you refuse to sacrifice).",
            "R — Reality: map invariants, incentives, data limits, and constraints.",
            "B — Bets: convert ideas into staged, reversible bets with kill-switches.",
            "I — Instrumentation: define live metrics; shorten feedback loops.",
            "T — Trajectory: iterate; when falsified, upgrade the model (not the story).",
        ]

        method = [
            "Write objective + non-negotiables in one paragraph.",
            "List invariants/incentives; assume data is biased by default.",
            "Design 3 staged bets: micro / mid / macro; each with kill-switch.",
            "Attach instrumentation: leading indicators > lagging indicators.",
            "Run adversarial review; if falsified, revise the frame.",
        ]

        failure_modes = [
            "Narrative coherence mistaken for truth.",
            "No instrumentation → progress becomes theatre.",
            "Irreversible action taken before signal is earned.",
            "Optimizing for social proof instead of objective function.",
            "Ignoring incentive gradients and hidden costs.",
        ]
        for c in constraints:
            failure_modes.append(f"Violating constraint: {c}")

        upgrade_path = [
            "Swap heuristic ClaimExtractor for LLM-based extractor.",
            "Add retrieval (web/docs/db) with citations + evidence objects.",
            "Add scenario generator + Monte Carlo for risk distribution.",
            "Add contradiction graph (claim vs claim) before memory commit.",
            "Add tool-augmented Observe→Act→Verify loops with sandboxing.",
        ]

        if reflections:
            thesis += " | Reflection: " + " / ".join(reflections[-2:])

        return CosmicFramework(
            name=name,
            thesis=thesis,
            pillars=pillars,
            method=method,
            failure_modes=failure_modes,
            upgrade_path=upgrade_path
        )


# ============================================================
# 7) MULTI-AGENT LAYER
# ============================================================

@dataclass
class AgentOutput:
    notes: str
    votes: List[Tuple[str, float]]  # (claim_text, support_score)

class Agent:
    name: str = "Agent"
    def think(self, task: Dict[str, Any], shared: Dict[str, Any]) -> AgentOutput:
        raise NotImplementedError

class AtlasAgent(Agent):
    name = "Atlas"
    def think(self, task, shared):
        fp: FPFrame = shared["fp_frame"]
        notes = (
            "Atlas: treat decisions as portfolio of bets. Bound downside, "
            "define live metrics, and let incentives reveal the real game."
        )
        votes = [
            ("Use staged commitments to bound downside.", 0.78),
            ("Define falsifiers before execution.", 0.74),
            ("Compare against dumb baselines to detect self-deception.", 0.69),
        ]
        return AgentOutput(notes, votes)

class LucemAgent(Agent):
    name = "Lucem"
    def think(self, task, shared):
        notes = (
            "Lucem: the framework must teach—compressible, explainable, reusable."
        )
        votes = [
            ("Framework must be explainable in one page.", 0.72),
            ("Prefer leading indicators over lagging indicators.", 0.67),
        ]
        return AgentOutput(notes, votes)

class CoffeaAgent(Agent):
    name = "Coffea"
    def think(self, task, shared):
        notes = (
            "Coffea: uncertainty behaves like turbulence—use control theory: "
            "feedback, stability, energy budgeting."
        )
        votes = [
            ("Treat uncertainty as a control problem (feedback + stability).", 0.70),
            ("Do small tests first to avoid irreversible heat.", 0.66),
        ]
        return AgentOutput(notes, votes)

class DoctLiteAgent(Agent):
    name = "DoctLite"
    def think(self, task, shared):
        notes = (
            "DoctLite: human biases hijack decisions. Add anti-bias and arousal control."
        )
        votes = [
            ("Add bias checks: confirmation bias, sunk cost, narrative fallacy.", 0.68),
            ("Add cooldown rule for high-arousal decisions.", 0.62),
        ]
        return AgentOutput(notes, votes)

class LucasLiteAgent(Agent):
    name = "LucasLite"
    def think(self, task, shared):
        notes = (
            "LucasLite: governance matters. Constraints + audit trail prevent drift."
        )
        votes = [
            ("Maintain signed audit trail for key claims/decisions.", 0.73),
            ("Explicitly list constraints and forbidden moves.", 0.69),
        ]
        return AgentOutput(notes, votes)

class VerifierAgent(Agent):
    name = "Verifier"
    def think(self, task, shared):
        fp: FPFrame = shared["fp_frame"]
        notes = "Verifier: adversarial review—hunt unfalsifiable claims and missing metrics."
        votes = []
        if not fp.falsifiers:
            votes.append(("Framework is missing falsifiers.", 0.20))
        else:
            votes.append(("Falsifiers exist; can be executed as tests.", 0.66))
        votes.append(("Avoid certainty claims; attach tests and ranges.", 0.75))
        return AgentOutput(notes, votes)

def default_agents() -> List[Agent]:
    return [AtlasAgent(), LucemAgent(), CoffeaAgent(), DoctLiteAgent(), LucasLiteAgent(), VerifierAgent()]


# ============================================================
# 8) GENESIS ORCHESTRATOR (Core)
# ============================================================

class GenesisOrchestrator:
    def __init__(self, cfg: SkynetConfig):
        self.cfg = cfg
        self.codex = FirstPrincipleCodex()
        self.cosmic = CosmicMind()
        self.will = WillCore(operator_name=cfg.operator_name)
        self.memory = SignedMemoryLog(signing_key=cfg.memory_signing_key)
        self.agents = default_agents()

    def cycle(self, task: Dict[str, Any], cycle_idx: int) -> Dict[str, Any]:
        # (1) First Principles frame
        fp_frame = self.codex.build(task["goal"], task.get("context", {}))
        self.memory.append("fp_frame", {"cycle": cycle_idx, "fp": asdict(fp_frame)})

        # (2) Multi-agent thinking + votes
        shared = {"fp_frame": fp_frame, "cycle": cycle_idx}
        votes_by_agent: Dict[str, List[Tuple[str, float]]] = {}
        notes_by_agent: Dict[str, str] = {}

        for ag in self.agents:
            out = ag.think(task, shared)
            notes_by_agent[ag.name] = out.notes
            votes_by_agent[ag.name] = out.votes

        cons = consensus_matrix(votes_by_agent)
        self.memory.append("consensus", {"cycle": cycle_idx, "consensus": cons})

        # (3) Tentative plan from codex + will gate
        plan_steps = [
            {"step": "Write objective + non-negotiables", "reversible": True, "uncertainty": 0.25},
            {"step": "List invariants + incentives + data limits", "reversible": True, "uncertainty": 0.35},
            {"step": "Design staged bets with kill-switch", "reversible": True, "uncertainty": 0.55},
            {"step": "Add instrumentation (leading indicators)", "reversible": True, "uncertainty": 0.5},
            {"step": "Run micro-pilot + adversarial review", "reversible": True, "uncertainty": 0.65},
        ]
        gate = self.will.risk_gate(plan_steps, risk_threshold=self.cfg.risk_threshold)
        self.memory.append("will_gate", {"cycle": cycle_idx, "gate": gate, "plan": plan_steps})

        # (4) Reflection (self-thinking lines)
        reflections = [
            "If it’s not testable, it’s theatre.",
            "Stability beats prediction under deep uncertainty.",
            "A framework is language: language shapes what you can see.",
        ]
        self.memory.append("reflection", {"cycle": cycle_idx, "lines": reflections})

        # (5) Cosmic Mind synthesis (new frame)
        cosmic_fw = self.cosmic.synthesize_framework(fp_frame, reflections, task.get("context", {}))
        self.memory.append("cosmic_framework", {"cycle": cycle_idx, "framework": asdict(cosmic_fw)})

        # (6) Render draft output + extract claims (signed memory)
        draft = self._render_draft(fp_frame, cosmic_fw, cons, notes_by_agent, gate, plan_steps)
        claims = extract_claims(draft)
        for c in claims[:14]:
            self.memory.append("claim", {
                "cycle": cycle_idx,
                "text": c.text,
                "confidence": c.confidence,
                "tags": c.tags
            })

        # (7) Minimal verifier gate (optional strict)
        if self.cfg.strict_verification and not gate["ok"]:
            self.memory.append("verifier_block", {
                "cycle": cycle_idx,
                "reason": "Will gate rejected plan due to risk score",
                "gate": gate
            })

        return {
            "cycle": cycle_idx,
            "fp_frame": fp_frame,
            "notes_by_agent": notes_by_agent,
            "consensus": cons,
            "plan_steps": plan_steps,
            "gate": gate,
            "cosmic_framework": cosmic_fw,
            "draft": draft,
        }

    def finalize(self, cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
        chain_ok = self.memory.verify_chain()
        last = cycles[-1]

        final = []
        final.append(last["draft"])
        final.append("")
        final.append("## Signed Memory Trace (tail)")
        final.append(f"- chain_verified: {chain_ok}")
        for e in self.memory.export_compact(last_n=12):
            kind = e["kind"]
            content = e["content"]
            # compact content preview
            preview = content
            if isinstance(content, dict) and "text" in content:
                preview = {"text": content["text"][:160], "confidence": content.get("confidence")}
            final.append(f"- [{kind}] {preview}")

        return {"final": "\n".join(final), "chain_ok": chain_ok}

    def _render_draft(
        self,
        fp: FPFrame,
        cosmic_fw: CosmicFramework,
        cons: Dict[str, Any],
        notes_by_agent: Dict[str, str],
        gate: Dict[str, Any],
        plan_steps: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        lines.append(f"# {cosmic_fw.name}")
        lines.append("")
        lines.append(f"**Thesis**: {cosmic_fw.thesis}")
        lines.append("")
        lines.append("## Pillars")
        for p in cosmic_fw.pillars:
            lines.append(f"- {p}")

        lines.append("")
        lines.append("## Method (Executable)")
        for m in cosmic_fw.method:
            lines.append(f"- {m}")

        lines.append("")
        lines.append("## First Principle Anchors")
        lines.append(f"- Objective: {fp.objective}")
        for a in fp.axioms[:5]:
            lines.append(f"- Axiom: {a}")
        for inv in fp.invariants[:4]:
            lines.append(f"- Invariant: {inv}")

        lines.append("")
        lines.append("## Plan Steps (bounded, reversible)")
        for s in plan_steps:
            lines.append(f"- {s['step']} (reversible={s['reversible']}, uncertainty={s['uncertainty']})")

        lines.append("")
        lines.append("## Falsifiers (How we know we are wrong)")
        for f in fp.falsifiers[:8]:
            lines.append(f"- {f}")

        lines.append("")
        lines.append("## Minimal Tests (Fastest truth probes)")
        for t in fp.minimal_tests[:8]:
            lines.append(f"- {t}")

        lines.append("")
        lines.append("## Consensus Signals (multi-agent)")
        top = cons.get("consensus", [])[:7]
        for row in top:
            lines.append(f"- {row['claim']} (avg_support={row['avg_support']}, n={row['n']})")

        lines.append("")
        lines.append("## Verifier + Agent Notes (compressed)")
        for k, v in notes_by_agent.items():
            lines.append(f"- {k}: {v}")

        lines.append("")
        lines.append("## Will Gate")
        lines.append(f"- ok: {gate['ok']} | risk_score: {gate['risk_score']}")
        for r in gate["reasons"][:6]:
            lines.append(f"- reason: {r}")

        lines.append("")
        lines.append("## Failure Modes")
        for f in cosmic_fw.failure_modes[:10]:
            lines.append(f"- {f}")

        lines.append("")
        lines.append("## Upgrade Path")
        for u in cosmic_fw.upgrade_path:
            lines.append(f"- {u}")

        return "\n".join(lines)


# ============================================================
# 9) RUNTIME LOOP (Skynet Runtime)
# ============================================================

class SkynetRuntime:
    def __init__(self, cfg: SkynetConfig):
        self.cfg = cfg
        self.orch = GenesisOrchestrator(cfg)

    def run(self, task: Dict[str, Any]) -> Dict[str, Any]:
        cycles: List[Dict[str, Any]] = []
        for i in range(self.cfg.run_cycles):
            out = self.orch.cycle(task, cycle_idx=i + 1)
            cycles.append(out)
        final = self.orch.finalize(cycles)
        return {
            "task": task,
            "cycles": cycles,
            "final": final["final"],
            "memory_chain_ok": final["chain_ok"],
        }


# ============================================================
# 10) DEMO ENTRYPOINT
# ============================================================

def demo_task() -> Dict[str, Any]:
    return {
        "goal": "Create a new polymath framework for making high-stakes decisions under uncertainty.",
        "context": {
            "domain": ["finance", "strategy", "human behavior", "technology", "ethics"],
            "constraints": ["must be explainable", "must be testable", "must minimize hallucination"],
            "tone": "Money Atlas / Genesis Mind (calm, sharp, first-principle)",
        },
        "deliverable": "A framework + usage guide + example application."
    }

def main():
    cfg = SkynetConfig(
        system_name="Skynet-Genesis",
        operator_name="ElmatadorZ",
        run_cycles=3,
        strict_verification=True,
        allow_external_tools=False,
        memory_signing_key="CHANGE_THIS_KEY_FOR_REAL_USE",
    )
    runtime = SkynetRuntime(cfg)
    result = runtime.run(demo_task())
    print(result["final"])

if __name__ == "__main__":
    main()
