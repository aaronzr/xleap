# AGENTS.md

Guidance for agents working in this repository.

## Scope

- Apply this guidance to all edits in this workspace unless a user request explicitly overrides it.
- Favor minimal, task-focused changes over broad rewrites.
- Preserve existing architecture and naming patterns unless the request requires a change.

## General Practice

- Read the relevant code before editing. Prefer matching existing patterns over introducing new structure.
- Keep changes focused on the request. Avoid opportunistic refactors, formatting churn, and unrelated cleanup.
- Treat the worktree as shared. Do not overwrite, revert, or delete changes you did not make unless explicitly asked.
- Prefer small, reviewable patches with clear names and straightforward control flow.
- Use structured parsing and existing helpers when available; avoid brittle string handling for structured data.
- Add comments only when they explain non-obvious intent or constraints.
- If requirements are ambiguous, choose the smallest safe interpretation and document assumptions in the final response.
- Before editing, identify the narrowest file set needed; do not touch unrelated files.

## Python

- Follow the style already present in nearby modules.
- Keep imports organized and remove unused code when it is part of your change.
- Prefer explicit error handling at I/O, network, archive, and UI boundaries.
- Avoid hidden global state unless the surrounding code already depends on it.
- Keep UI work responsive. Long-running archive fetches, plotting work, or data processing should not block the Qt event loop.
- Prefer pure helper functions for data transforms and keep side effects near call boundaries.
- Preserve type hints in typed modules; add or refine hints when touching related code.

## Notebooks

- Keep notebook changes minimal and intentional.
- Put reusable logic in Python modules under `sparklines/app/` rather than duplicating it across notebooks.
- Avoid committing large generated outputs unless they are necessary for the task.
- Keep notebook cells readable and ordered: imports, configuration, helpers, execution/UI.
- For workshop notebooks, prioritize self-contained examples with explicit dependencies.

## Testing

- Run the narrowest useful tests for the change, then broader tests when touching shared behavior.
- Prefer `uv run pytest` for the full test suite when dependencies are available.
- For targeted checks, run commands such as:

```bash
uv run pytest tests/test_sparklines_plot_utils.py
uv run pytest tests/test_sparklines_hierarchy.py
```

- If tests cannot be run, state the blocker clearly in the final response.
- For notebook-only changes, run at least the affected cells when feasible and report what was executed.

## Verification

- For plotting or UI changes, verify both behavior and visual impact where practical.
- For archive or time-range changes, test representative empty, partial, and populated data cases.
- For config changes, validate YAML syntax and confirm file paths are resolved from the expected working directory.
- Validate failure paths, not only happy paths (network failures, empty payloads, invalid user input).

## Communication

- Summarize what changed, why, and how it was validated.
- Include concrete file references and mention any untested or risky areas.
- If blocked, report the exact blocker and the next best actionable step.

## Git Hygiene

- Check `git status` before and after significant edits.
- Do not commit unless the user asks.
- Keep generated caches, logs, and local artifacts out of commits unless explicitly needed.
