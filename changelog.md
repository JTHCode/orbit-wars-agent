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

## 2026-05-21 — Add phase hysteresis, guardrails, emergency defense, and transition diagnostics

- Summary: Implemented `select_phase_with_hysteresis(...)` with margin/persistence gates, endgame and pressure guardrails, emergency defense mode signaling, and rich transition diagnostics in runtime stats; wired `phase()` through the selector and regenerated `main.py`.
- Why: Reduce phase flapping, prevent premature/unsafe strategic mode shifts, and improve offline replay analysis of phase transitions.
- Impact: More stable strategic phase control with explicit override and debug metadata; no API/signature changes.
- Files: `orbit_agent/core.py`, `main.py`, `changelog.md`

## 2026-05-21 — Add agent-maintained changelog and expand agent guidance

- Summary: Added `changelog.md` and expanded `agents.md` with project-specific guidance for modular development, submission build workflow, testing, and maintenance expectations.
- Why: Improve multi-agent coordination, reduce mistakes, and preserve project context across iterations.
- Impact: Documentation/workflow only; no bot behavior changes.
- Files: `agents.md`, `changelog.md`
