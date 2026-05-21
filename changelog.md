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

## 2026-05-21 — Add phase diagnostics telemetry and benchmark aggregations

- Summary: Exposed phase telemetry via `get_agent_stats` (current phase, per-phase scores, normalized signals, hysteresis counters, transition events with reason bits) and extended benchmark summaries with phase occupancy buckets, transition counts per game, and win-rate deltas by dominant midgame phase trajectory.
- Why: Enable iterative calibration of phase score weights and transition thresholds from reproducible telemetry rather than anecdotal game inspection.
- Impact: Adds richer debug/stats surfaces and benchmark analytics; no changes to submission interface.
- Files: `orbit_agent/core.py`, `orbit_agent/benchmark.py`, `main.py`, `changelog.md`

---

## 2026-05-21 — Add agent-maintained changelog and expand agent guidance

- Summary: Added `changelog.md` and expanded `agents.md` with project-specific guidance for modular development, submission build workflow, testing, and maintenance expectations.
- Why: Improve multi-agent coordination, reduce mistakes, and preserve project context across iterations.
- Impact: Documentation/workflow only; no bot behavior changes.
- Files: `agents.md`, `changelog.md`
