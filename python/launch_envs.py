#!/usr/bin/env python3
"""Launch multiple ForgeEnvServer instances on sequential ports.

Usage:
  /home/ddroder/users/dan/forge_env/.venv/bin/python launch_envs.py --n 4 --base-port 8799

This script starts Java processes in the background and waits for /health.
It writes PIDs to stdout for convenience.

NOTE: This is a minimal launcher. You can still stop them with pkill:
  pkill -f 'java.*rl.ForgeEnvServer'
"""

from __future__ import annotations

import argparse
import subprocess
import time

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=8799)
    ap.add_argument("--xmx", type=str, default="6G")
    args = ap.parse_args()

    n = max(1, args.n)
    base = args.base_port

    jar = "/home/ddroder/users/dan/forge_env/rl_v0/build/forge-env-server.jar"
    forgejar = "/home/ddroder/users/dan/forge_env/forge_dist/forge-gui-desktop-2.0.09-jar-with-dependencies.jar"
    jre = "/home/ddroder/users/dan/forge_env/jre17/bin/java"

    procs: list[tuple[int, subprocess.Popen]] = []

    for i in range(n):
        port = base + i
        cmd = [
            jre,
            f"-Xmx{args.xmx}",
            "-Djava.awt.headless=true",
            "-Dforge.assetsDir=/home/ddroder/users/dan/forge_env/forge_dist/",
            f"-Dforge.port={port}",
            "-cp",
            f"{jar}:{forgejar}",
            "rl.ForgeEnvServer",
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append((port, p))

    # wait for readiness
    t0 = time.time()
    for port, p in procs:
        url = f"http://127.0.0.1:{port}/health"
        while time.time() - t0 < 60:
            try:
                j = requests.get(url, timeout=1).json()
                if j.get("ok") is True:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError(f"env on port {port} did not become healthy")

    for port, p in procs:
        print(f"port={port} pid={p.pid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
