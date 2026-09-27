"""src/tunable.py is the mechanism by which tools/tune_thresholds.py's
output takes effect with zero source-code edits (see src/rules.py,
src/risk.py). Its safety property -- a tuned_params.json can only retune an
existing constant, never inject a new attribute or override a denylisted
one -- is worth a direct unit test, independent of whichever module happens
to use it.
"""
import json

import src.tunable as tunable_mod
from src.tunable import apply_overrides


def _point_at(monkeypatch, tmp_path, content):
    """`_PATH` is computed once from the env var at module-import time (by
    design -- production code shouldn't re-read an env var on every call),
    so tests must patch the module-level `_PATH`/`_cache` directly rather
    than set the env var after the fact.
    """
    params_path = tmp_path / "tuned_params.json"
    if content is not None:
        params_path.write_text(json.dumps(content))
    monkeypatch.setattr(tunable_mod, "_PATH", str(params_path))
    monkeypatch.setattr(tunable_mod, "_cache", None)


def test_override_only_touches_existing_keys(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path, {"demo": {"KNOWN": 42, "INJECTED_NEW_NAME": 999}})

    target = {"KNOWN": 1}
    apply_overrides(target, "demo")

    assert target["KNOWN"] == 42
    assert "INJECTED_NEW_NAME" not in target


def test_deny_list_is_enforced_even_if_key_exists(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path, {"demo": {"H_SEC": 999.0}})

    target = {"H_SEC": 5.0}
    apply_overrides(target, "demo", deny=frozenset({"H_SEC"}))

    assert target["H_SEC"] == 5.0, "denylisted constants must never be overridden"


def test_missing_tuned_params_file_is_a_silent_noop(tmp_path, monkeypatch):
    _point_at(monkeypatch, tmp_path, content=None)  # file is never written

    target = {"SOMETHING": 1}
    apply_overrides(target, "demo")
    assert target["SOMETHING"] == 1
