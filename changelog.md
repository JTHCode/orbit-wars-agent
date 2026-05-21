# Changelog

This changelog is maintained by agents working in this repository.

## How to use this file

- Add a new entry for every meaningful code, architecture, workflow, or documentation change.
- Keep entries concise and objective; include what changed and why.
- Prefer newest entries at the top (reverse chronological order).
- Reference impacted files when helpful.

## Entry template

```md
## YYYY-MM-DD — <short title>

- Summary: <what changed>
- Why: <reason/motivation>
- Impact: <runtime/behavior/workflow effects>
- Files: `<path1>`, `<path2>`
```

---


## 2026-05-21 — Wire unused phase config knobs and add mode hysteresis

- Summary: Hooked `PHASE_CONFIG.score_weights` into phase score computation, added hysteresis/persistence for strategic mode (`ahead/even/behind`), and removed stale unused phase constants left over from turn-gated policy.
- Why: Close remaining gaps from the state-driven migration so declared config knobs actually control live behavior and reduce hidden mode flicker.
- Impact: Phase and mode switching are now more stable and tunable through `PHASE_CONFIG`; less dead/duplicated policy state remains in code.
- Files: `orbit_agent/core.py`, `main.py`, `changelog.md`

## 2026-05-21 — Activate state-driven phases by default and de-hardcode policy gates

- Summary: Enabled state-driven phases by default, wired `choose_phase()` hysteresis/guardrail values to `PHASE_CONFIG`, and replaced remaining step-only phase gates with phase-aware checks (including nearest-danger and pressure-opportunity gating).
- Why: Complete the transition from legacy turn buckets to the state-driven phase system and prevent config drift from hardcoded duplicate thresholds.
- Impact: Active runtime phase policy now follows state signals/hysteresis by default; phase tuning is now centralized in `PHASE_CONFIG` and less brittle across modules.
- Files: `orbit_agent/core.py`, `main.py`, `agents.md`, `changelog.md`

## 2026-05-21 — Sync phase-policy docs and benchmark comparison guidance

- Summary: Updated project docs to explicitly reference the new state-driven phase pipeline (`compute_phase_signals` → `compute_phase_scores` → `choose_phase`), hysteresis/override hooks, and rollout flag ownership; added benchmark guidance for controlled old-vs-new phase-policy comparisons.
- Why: Ensure future agent changes are anchored to the correct source-of-truth phase functions and evaluated with apples-to-apples benchmark methodology.
- Impact: Documentation/workflow only; no runtime behavior changes.
- Files: `agents.md`, `changelog.md`

## 2026-05-21 — Add phase debug telemetry and guarded rollout flags

- Summary: Added lightweight phase telemetry through `get_agent_stats()` including active phase, phase scores, core normalized signals, hysteresis counters, override flags, and last transition metadata; introduced rollout flags for state-driven phases and optional shadow comparison mode.
- Why: Enable fast tuning/diagnostics for thrash and mistimed phase shifts while keeping a conservative default rollout path.
- Impact: Default runtime now keeps legacy turn-bound phase activation (`USE_STATE_DRIVEN_PHASES = False`) while still computing state-driven scores for debug/transition insight; optional flags enable live transition logging and shadow checks.
- Files: `orbit_agent/core.py`, `main.py`, `changelog.md`

## 2026-05-21 — Replace fixed turn phase with score-driven phase state

- Summary: Reworked `World.phase()` to use score-based phase selection from `self.phase_signals`, with margin, persistence, minimum hold time, and an ETA-aware mandatory `final_scoring` lock; also added an emergency defense override flag driven by threatened-owned-value.
- Why: Reduce phase thrashing, make phase transitions responsive to board state instead of fixed turns, and allow defensive tactics to react immediately under high threat.
- Impact: Strategic phase now adapts to game state each turn while preserving legacy phase string compatibility; tactical code can key off `modes["emergency_defense"]` / `phase_overrides["emergency_defense"]` for forced defense posture.
- Files: `orbit_agent/core.py`, `main.py`, `changelog.md`

## 2026-05-21 — Add agent-maintained changelog and expand agent guidance

- Summary: Added `changelog.md` and expanded `agents.md` with project-specific guidance for modular development, submission build workflow, testing, and maintenance expectations.
- Why: Improve multi-agent coordination, reduce mistakes, and preserve project context across iterations.
- Impact: Documentation/workflow only; no bot behavior changes.
- Files: `agents.md`, `changelog.md`
