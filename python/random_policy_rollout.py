#!/usr/bin/env python3
"""v0 trainer: run episodes with a dumb policy and log to TensorBoard.

This is NOT PPO yet.
It proves:
- reset/wait_ready loop
- action selection from legal_actions
- reward/done handling from env
- TensorBoard logging

Run:
  /home/ddroder/users/dan/forge_env/.venv/bin/python random_policy_rollout.py --episodes 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests
from tensorboardX import SummaryWriter

BASE = "http://127.0.0.1:8799"


def choose_action(actions: list[str]) -> str:
    for a in ("PLAY_FIRST_LAND", "CAST_FIRST_SPELL", "ATTACK_ALL", "BLOCK_NONE"):
        if a in actions:
            return a
    if "USE_AI_COMBAT" in actions:
        return "USE_AI_COMBAT"
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--logdir", default=str(Path(__file__).resolve().parents[1] / "runs" / "rl_v0_random"))
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--max_turn", type=int, default=60)
    args = ap.parse_args()

    writer = SummaryWriter(logdir=args.logdir)

    for ep in range(1, args.episodes + 1):
        requests.post(BASE + "/reset", timeout=10)
        ready = requests.get(BASE + "/wait_ready", timeout=30).json()

        ep_start = time.time()
        steps = 0
        last = ready
        ep_reward = 0.0
        timed_out = False

        for _ in range(args.max_steps):
            steps += 1

            # safety valve: turn cap => draw
            turn = last.get("turn")
            if isinstance(turn, int) and turn >= args.max_turn:
                timed_out = True
                break

            la = requests.get(BASE + "/legal_actions", timeout=10).json()
            actions = la.get("actions", ["PASS"])
            act = choose_action(actions)
            last = requests.post(BASE + "/step", json={"action": act}, timeout=30).json()

            writer.add_scalar("step/p1_life", (last.get("p1_life") or 0), (ep * 100000) + steps)
            writer.add_scalar("step/p2_life", (last.get("p2_life") or 0), (ep * 100000) + steps)
            writer.add_scalar("step/turn", (last.get("turn") or 0), (ep * 100000) + steps)
            writer.add_text("step/phase", str(last.get("phase")), (ep * 100000) + steps)

            if last.get("done"):
                ep_reward = float(last.get("reward") or 0.0)
                break

        if timed_out:
            ep_reward = 0.0

        dur = time.time() - ep_start
        writer.add_scalar("episode/reward", ep_reward, ep)
        writer.add_scalar("episode/steps", steps, ep)
        writer.add_scalar("episode/seconds", dur, ep)
        writer.add_scalar("episode/timed_out", 1 if timed_out else 0, ep)
        writer.add_text("episode/winner", str(last.get("winner")), ep)

        print(
            f"episode {ep}: reward={ep_reward} steps={steps} timed_out={timed_out} winner={last.get('winner')} "
            f"turn={last.get('turn')} p1={last.get('p1_life')} p2={last.get('p2_life')}"
        )

        # small pause so the watcher threads settle
        time.sleep(0.2)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
