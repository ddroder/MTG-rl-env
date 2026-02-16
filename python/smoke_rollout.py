#!/usr/bin/env python3
"""Fast smoke test: run a handful of env decisions and report throughput.

This is *not* training. It just verifies that:
- /reset works
- we can repeatedly reach waiting points
- /act returns
- a few dozen decisions complete within a time budget

Run:
  /home/ddroder/users/dan/forge_env/.venv/bin/python smoke_rollout.py
"""

from __future__ import annotations

import sys
import time

import requests

BASE = "http://127.0.0.1:8799"


def main() -> int:
    requests.post(BASE + "/reset", timeout=10)

    decisions = 0
    steps = 0
    t0 = time.time()
    last = None

    while decisions < 25 and (time.time() - t0) < 60:
        last = requests.post(BASE + "/advance", timeout=15).json()
        steps += 1
        if last.get("done"):
            break
        if last.get("waiting"):
            la = requests.get(BASE + "/legal_actions", timeout=10).json().get("actions", ["PASS"])
            # pick first non-PASS action if present
            action_id = 1 if len(la) > 1 else 0
            a = requests.post(BASE + "/act", json={"action_id": action_id}, timeout=20).json()
            if not a.get("ok"):
                raise RuntimeError(f"/act failed: {a}")
            decisions += 1

    dt = time.time() - t0
    print(f"decisions={decisions} advance_calls={steps} seconds={dt:.2f} decisions_per_sec={decisions/max(dt,1e-6):.3f}")
    if last:
        print("last:", {k: last.get(k) for k in ["turn", "phase", "priority_player", "waiting", "done"]})

    if decisions < 5:
        raise SystemExit("too few decisions; env likely stalled")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("SMOKE FAILED:", e, file=sys.stderr)
        raise SystemExit(2)
