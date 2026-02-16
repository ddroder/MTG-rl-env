#!/usr/bin/env python3
"""Preflight checks for the Forge RL env.

Goal: catch common failure modes *before* starting long training runs.

Checks:
- server health endpoint responds
- /reset + /wait_ready yield a non-null phase/turn
- state contains expected feature keys
- we can reach a main phase within a reasonable number of PASS steps
- legal_actions in a main phase contains at least PASS + something else (ideally a land)
- /advance eventually reaches a decision point (waiting=true)
- quick stepping loop completes a few decisions within a time budget

Run:
  /home/ddroder/users/dan/forge_env/.venv/bin/python rl_preflight.py
"""

from __future__ import annotations

import sys
import time
from typing import Any

import requests

import os
BASE = os.environ.get("FORGE_BASE_URL") or f"http://127.0.0.1:{os.environ.get('FORGE_BASE_PORT','8799')}"


def req(method: str, path: str, **kwargs) -> dict[str, Any]:
    timeout = kwargs.pop("timeout", 10)
    last_err: Exception | None = None
    for _ in range(3):
        try:
            r = requests.request(method, BASE + path, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    raise last_err  # type: ignore[misc]


def main() -> int:
    # 1) health
    h = req("GET", "/health")
    assert h.get("ok") is True, f"/health not ok: {h}"

    # 2-4) reset + wait_ready + basic sanity + reach main phase.
    max_resets = 5
    for attempt in range(1, max_resets + 1):
        req("POST", "/reset")
        ready = req("GET", "/wait_ready", timeout=30)
        assert ready.get("ok") is True, f"/wait_ready not ok: {ready}"
        assert ready.get("turn") is not None, f"turn missing: {ready.keys()}"
        assert ready.get("phase") is not None, f"phase missing: {ready.keys()}"

        expected_keys = [
            "p1_hand_names",
            "p1_untapped_lands",
            "p1_commander_zone",
            "p1_commander_cast",
        ]
        missing = [k for k in expected_keys if k not in ready]
        assert not missing, f"missing keys in state: {missing}"

        hn = ready.get("p1_hand_names")
        assert isinstance(hn, list) and len(hn) == 10, f"p1_hand_names should be list len 10, got: {type(hn)} {hn}"

        reached_main = False
        for i in range(800):
            st = req("GET", "/wait_ready")
            ph = st.get("phase")
            if ph in ("MAIN1", "MAIN2"):
                reached_main = True
                la = req("GET", "/legal_actions")
                acts = la.get("actions", [])
                assert isinstance(acts, list) and len(acts) >= 1, f"bad actions: {la}"
                if len(acts) > 1:
                    print(
                        f"OK: reached {ph} in {i} steps (reset attempt {attempt}); legal_actions={acts[:12]} (len={len(acts)})",
                        flush=True,
                    )
                    break
                # PASS-only: keep advancing and keep searching for MAIN with real actions
                req("POST", "/advance_wait", timeout=10)
                continue
            req("POST", "/step", json={"action_id": 0}, timeout=10)
            time.sleep(0.005)

        if not reached_main:
            raise AssertionError("Never reached MAIN phase within step budget")

        la = req("GET", "/legal_actions")
        acts = la.get("actions", [])
        if isinstance(acts, list) and len(acts) > 1:
            break
        # If we're in a PASS-only window, try to advance via /advance_wait and re-check.
        for _ in range(50):
            req("POST", "/advance_wait", timeout=10)
            la2 = req("GET", "/legal_actions")
            acts2 = la2.get("actions", [])
            if isinstance(acts2, list) and len(acts2) > 1:
                break
        else:
            # keep trying other resets
            continue
        break
    else:
        raise AssertionError("Reached MAIN but only PASS was legal across multiple resets")

    # 5) /advance should eventually reach a decision point (waiting=true).
    seen_waiting = False
    for _ in range(300):
        r = req("POST", "/advance", timeout=10)
        if r.get("waiting") is True:
            seen_waiting = True
            break
        time.sleep(0.01)
    assert seen_waiting, "Never observed waiting=true from /advance (possible deadlock)"

    # 6) Quick stepping loop: complete a few decisions.
    t0 = time.time()
    decisions = 0
    for _ in range(60):
        r = req("POST", "/advance", timeout=10)
        if r.get("done") is True:
            break
        if r.get("waiting") is True:
            # Prefer the first non-PASS action if present.
            la = req("GET", "/legal_actions", timeout=10)
            acts = la.get("actions", ["PASS"])
            action_id = 1 if isinstance(acts, list) and len(acts) > 1 else 0
            a = req("POST", "/act", json={"action_id": int(action_id)}, timeout=30)
            assert a.get("ok") is True, f"/act failed: {a}"
            decisions += 1
        time.sleep(0.01)

    dt = time.time() - t0
    assert decisions >= 3, f"preflight only got {decisions} decisions; stepping may be broken"
    assert dt < 45.0, f"env responsiveness too slow ({decisions} decisions took {dt:.1f}s)"

    print("Preflight OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("Preflight FAILED:", e, file=sys.stderr)
        raise SystemExit(2)
