#!/usr/bin/env python3
"""Benchmark parallel Forge env throughput.

Method:
- For n_envs in 1..max
  - kill existing train_ppo + ForgeEnvServer
  - launch n_envs servers on sequential ports
  - run train_ppo.py in DEBUG_FAST mode for BENCH_TOTAL_TIMESTEPS (default 4096)
  - read TensorBoard scalar time/fps from the newest run directory
  - print a compact summary and a resource snapshot

Usage:
  /home/ddroder/users/dan/forge_env/.venv/bin/python bench_parallel.py --max-envs 4
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sh(cmd: str, timeout: int = 1200) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, timeout=timeout)


def newest_run() -> str | None:
    runs = sorted(glob.glob('/home/ddroder/users/dan/forge_env/rl_v0/runs/ppo_v0/MaskablePPO_*'), key=os.path.getmtime)
    return runs[-1] if runs else None


def read_fps(run_dir: str) -> tuple[float | None, int | None, int | None]:
    files = sorted(glob.glob(run_dir + '/events.out.tfevents.*'), key=os.path.getmtime)
    if not files:
        return None, None, None
    path = files[-1]
    size = os.path.getsize(path)
    ea = EventAccumulator(path)
    ea.Reload()
    tags = set(ea.Tags().get('scalars', []))
    if 'time/fps' not in tags:
        return None, size, None
    s = ea.Scalars('time/fps')
    return float(s[-1].value), size, int(s[-1].step)


def run_one(n_envs: int, base_port: int, bench_ts: int) -> dict:
    subprocess.call("pkill -f 'python.*train_ppo.py'", shell=True)
    subprocess.call("pkill -f 'java.*rl.ForgeEnvServer'", shell=True)
    time.sleep(0.5)

    sh(
        f"/home/ddroder/users/dan/forge_env/.venv/bin/python /home/ddroder/users/dan/forge_env/rl_v0/python/launch_envs.py --n {n_envs} --base-port {base_port}",
        timeout=180,
    )

    env = os.environ.copy()
    env['DEBUG_FAST'] = '1'
    env['FORGE_N_ENVS'] = str(n_envs)
    env['FORGE_BASE_PORT'] = str(base_port)
    env['BENCH_TOTAL_TIMESTEPS'] = str(bench_ts)

    before = newest_run()

    p = subprocess.Popen(
        [
            '/home/ddroder/users/dan/forge_env/.venv/bin/python',
            '-u',
            '/home/ddroder/users/dan/forge_env/rl_v0/python/train_ppo.py',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )

    # wait for a new run dir to appear
    t0 = time.time()
    run_dir = None
    while time.time() - t0 < 180:
        cur = newest_run()
        if cur and cur != before:
            run_dir = cur
            break
        time.sleep(0.5)

    # wait for fps scalar
    fps = None
    step = None
    size = None
    if run_dir:
        for _ in range(60):
            fps, size, step = read_fps(run_dir)
            if fps is not None:
                break
            time.sleep(1)

    # stop trainer
    p.terminate()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()

    snap = sh("ps -o pid,%cpu,%mem,rss,cmd -C java | egrep 'rl.ForgeEnvServer' || true")

    return {
        'n_envs': n_envs,
        'fps': fps,
        'tb_step': step,
        'tfevents_size': size,
        'run': os.path.basename(run_dir) if run_dir else None,
        'java': snap.strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-envs', type=int, default=4)
    ap.add_argument('--base-port', type=int, default=8799)
    ap.add_argument('--bench-ts', type=int, default=4096)
    args = ap.parse_args()

    results = []
    for n in range(1, args.max_envs + 1):
        print(f"\n=== bench n_envs={n} ===", flush=True)
        r = run_one(n, args.base_port, args.bench_ts)
        results.append(r)
        print(f"n={n} run={r['run']} fps={r['fps']} step={r['tb_step']} eventsz={r['tfevents_size']}")

    print("\nSUMMARY")
    for r in results:
        print(f"n={r['n_envs']} fps={r['fps']} run={r['run']}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
