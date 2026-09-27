"""Loads tuned constant overrides from tuned_params.json, if present.

`tools/tune_thresholds.py` grid-searches the module-level constants in
src/rules.py and src/risk.py against a real annotated dev set and writes its
best combination here. Both modules call `apply_overrides(globals(), group)`
once, at import time, right after defining their defaults -- this is what
lets a tuning run take effect with zero source-code edits: overwrite
tuned_params.json (or point GENZ_TUNED_PARAMS at a different file) and
re-import.

Until a real dev set exists and the tool has actually been run, no
tuned_params.json ships in the repo, so every constant is exactly the
hand-picked default documented in README.md / docs/REPORT.md.
"""
from __future__ import annotations

import json
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PATH = os.environ.get("GENZ_TUNED_PARAMS", os.path.join(_REPO_ROOT, "tuned_params.json"))

_cache = None


def _load():
    global _cache
    if _cache is None:
        if os.path.isfile(_PATH):
            with open(_PATH, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def apply_overrides(module_globals: dict, group: str, deny: frozenset = frozenset()) -> None:
    """Only overwrites names that already exist as constants in the calling
    module -- a tuned_params.json can never inject a new attribute, only
    retune an existing, documented one. `deny` is a hard exclusion list
    (e.g. spec-fixed constants like H_SEC) enforced here, not left to the
    tuner's grid to simply not include -- so a hand-edited tuned_params.json
    can't accidentally break a value the spec fixes.
    """
    overrides = _load().get(group, {})
    for key, value in overrides.items():
        if key in deny:
            continue
        if key in module_globals:
            module_globals[key] = value
