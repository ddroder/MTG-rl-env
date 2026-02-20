#!/usr/bin/env python3
"""Train PPO against the Forge HTTP env (action_id-based).

Run:
  /home/ddroder/users/dan/forge_env/.venv/bin/python train_ppo.py

TensorBoard:
  tensorboard --logdir /home/ddroder/users/dan/forge_env/rl_v0/runs
"""

from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from forge_gym_env import ForgeHttpEnv


class ForgeExtraEvalMetricsCallback(BaseCallback):
    """Logs eval win_rate and invalid_action_rate to TensorBoard.

    Also optionally saves per-eval-episode replays:
    - Forge GameLog dump (text)
    - RL action trace (jsonl)

    This runs short deterministic eval rollouts on a non-Vec env so we can read
    info['winner'] and info['invalid_action'].
    """

    def __init__(
        self,
        base_url: str,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 10,
        verbose: int = 0,
        record_replays: str = "off",  # off|eval
        replay_dir: str | None = None,
    ):
        super().__init__(verbose)
        self.base_url = str(base_url).rstrip("/")
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.record_replays = str(record_replays)
        self.replay_dir = replay_dir
        self._eval_ep_counter = 0

    def _ensure_replay_dir(self):
        if self.record_replays.lower() == "off":
            return None
        from pathlib import Path
        if self.replay_dir:
            d = Path(self.replay_dir)
        else:
            # Try to derive from SB3 logger dir
            d0 = getattr(self.logger, "dir", None)
            if d0 is None and getattr(self, "model", None) is not None:
                d0 = getattr(getattr(self.model, "logger", None), "dir", None)
            if d0 is None:
                return None
            d = Path(str(d0)) / "replays"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        last = getattr(self, "_last_eval_timesteps", 0)
        if (self.num_timesteps - last) < self.eval_freq:
            return True
        self._last_eval_timesteps = int(self.num_timesteps)

        # For this callback we want a *non-Vec* env.
        # ActionMasker passes the wrapped env (here: Monitor(...)) into mask_fn,
        # so we must reach through to the underlying ForgeHttpEnv.
        def mask_fn(e):
            base = getattr(e, "unwrapped", e)
            return base.action_masks()

        env = ActionMasker(Monitor(ForgeHttpEnv(base_url=self.base_url)), mask_fn)

        wins = 0
        invalids = 0
        timeouts = 0
        total_steps = 0
        lengths = []
        dmg_opp_eps = []
        dmg_self_eps = []
        spells_cast_eps = []
        activations_eps = []
        lands_eps = []
        commander_eps = []
        commander_legal_eps = []
        commander_any_eps = []
        commander_rate_eps = []
        our_lost_eps = []
        opp_lost_eps = []

        max_steps_per_episode = 400  # safety cap to avoid eval hanging forever

        replay_root = self._ensure_replay_dir()

        for _ in range(self.n_eval_episodes):
            obs, info = env.reset()
            done = False
            ep_len = 0

            trace_lines = []

            while not done and ep_len < max_steps_per_episode:
                mask = env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = bool(terminated or truncated)
                ep_len += 1
                total_steps += 1
                if info.get("invalid_action"):
                    invalids += 1

                if replay_root is not None and self.record_replays.lower() == "eval":
                    # best-effort pull turn/phase from underlying ForgeHttpEnv
                    st = None
                    try:
                        base_env = env
                        for _ in range(6):
                            base_env = getattr(base_env, "env", None) or getattr(base_env, "unwrapped", base_env)
                        st = getattr(base_env, "_last_state", None)
                    except Exception:
                        st = None

                    trace_lines.append({
                        "t": ep_len,
                        "turn": (st or {}).get("turn"),
                        "phase": (st or {}).get("phase_name") or (st or {}).get("phase"),
                        "p1_life": info.get("p1_life"),
                        "p2_life": info.get("p2_life"),
                        "action_taken": info.get("action_taken"),
                        "n_actions": len(info.get("actions") or []),
                    })

            # If we hit the cap, treat it as a truncated eval episode
            if not done and ep_len >= max_steps_per_episode:
                timeouts += 1
                done = True

            # Save replay artifacts (eval only)
            if replay_root is not None and self.record_replays.lower() == "eval":
                from datetime import datetime
                import json
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                ep_id = self._eval_ep_counter
                self._eval_ep_counter += 1

                # action trace
                trace_path = replay_root / f"eval_ep_{ep_id:06d}_{ts}_trace.jsonl"
                with open(trace_path, "w", encoding="utf-8") as f:
                    for row in trace_lines:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                # Forge GameLog
                try:
                    import requests
                    gl = requests.get(self.base_url + "/gamelog", timeout=5).json()
                    txt = gl.get("text") or ""
                    gp = replay_root / f"eval_ep_{ep_id:06d}_{ts}_gamelog.txt"
                    with open(gp, "w", encoding="utf-8") as f:
                        f.write(txt)
                except Exception:
                    pass

            lengths.append(ep_len)
            dmg_opp_eps.append(float(info.get("ep_dmg_to_opp", 0.0)))
            dmg_self_eps.append(float(info.get("ep_dmg_to_self", 0.0)))
            spells_cast_eps.append(float(info.get("ep_spells_cast", 0.0)))
            activations_eps.append(float(info.get("ep_activations_used", 0.0)))
            lands_eps.append(float(info.get("ep_lands_played", 0.0)))
            commander_eps.append(float(info.get("ep_commander_casts", 0.0)))
            commander_legal_eps.append(float(info.get("ep_commander_legal_steps", 0.0)))
            commander_any_eps.append(float(info.get("ep_commander_in_actions_any", 0.0)))
            steps = float(info.get("ep_steps", ep_len))
            commander_rate_eps.append(float(info.get("ep_commander_in_actions_any", 0.0)) / max(1.0, steps))
            our_lost_eps.append(float(info.get("ep_our_creatures_lost", 0.0)))
            opp_lost_eps.append(float(info.get("ep_opp_creatures_lost", 0.0)))

            winner = info.get("winner")
            if isinstance(winner, str) and "External" in winner:
                wins += 1

        win_rate = wins / max(1, self.n_eval_episodes)
        invalid_rate = invalids / max(1, total_steps)
        timeout_rate = timeouts / max(1, self.n_eval_episodes)
        mean_len = sum(lengths) / max(1, len(lengths))

        # Write to TB
        self.logger.record("eval/win_rate", float(win_rate))
        self.logger.record("eval/invalid_action_rate", float(invalid_rate))
        self.logger.record("eval/timeout_rate", float(timeout_rate))
        self.logger.record("eval/mean_ep_length_det", float(mean_len))
        self.logger.record("eval/ep_dmg_to_opp", float(sum(dmg_opp_eps) / max(1, len(dmg_opp_eps))))
        self.logger.record("eval/ep_dmg_to_self", float(sum(dmg_self_eps) / max(1, len(dmg_self_eps))))
        self.logger.record("eval/ep_spells_cast", float(sum(spells_cast_eps) / max(1, len(spells_cast_eps))))
        self.logger.record("eval/ep_activations_used", float(sum(activations_eps) / max(1, len(activations_eps))))
        self.logger.record("eval/ep_lands_played", float(sum(lands_eps) / max(1, len(lands_eps))))
        self.logger.record("eval/ep_commander_casts", float(sum(commander_eps) / max(1, len(commander_eps))))
        self.logger.record("eval/ep_commander_legal_steps", float(sum(commander_legal_eps) / max(1, len(commander_legal_eps))))
        self.logger.record("eval/ep_commander_in_actions_any", float(sum(commander_any_eps) / max(1, len(commander_any_eps))))
        self.logger.record("eval/commander_offered_rate", float(sum(commander_rate_eps) / max(1, len(commander_rate_eps))))
        self.logger.record("eval/ep_our_creatures_lost", float(sum(our_lost_eps) / max(1, len(our_lost_eps))))
        self.logger.record("eval/ep_opp_creatures_lost", float(sum(opp_lost_eps) / max(1, len(opp_lost_eps))))
        return True


def main():
    import os, sys, time

    debug_fast = os.environ.get("DEBUG_FAST", "0") == "1"

    print(f"[train_ppo] starting (DEBUG_FAST={debug_fast})", flush=True)
    print(f"[train_ppo] FORGE_BASE_PORT={os.environ.get('FORGE_BASE_PORT')} FORGE_BASE_URL={os.environ.get('FORGE_BASE_URL')}", flush=True)

    # Wait for server to be fully ready (avoid races on restart)
    import requests

    def _json_get(url: str, timeout: float, retries: int = 10):
        import time as _t
        last = None
        for i in range(max(1, retries)):
            try:
                return requests.get(url, timeout=timeout).json()
            except Exception as e:
                last = e
                _t.sleep(0.05 + 0.05 * i)
        raise last

    base_port = int(os.environ.get("FORGE_BASE_PORT", "8799"))
    base = os.environ.get("FORGE_BASE_URL") or f"http://127.0.0.1:{base_port}"
    base = base.rstrip("/")
    t_ready0 = time.time()
    while time.time() - t_ready0 < 60:
        try:
            h = _json_get(base + "/health", timeout=2, retries=5)
            if h.get("ok") is True:
                # ensure a game is started
                try:
                    requests.post(base + "/reset", timeout=5)
                except Exception:
                    pass
                # also require wait_ready to respond with turn/phase
                s = _json_get(base + "/wait_ready", timeout=5, retries=5)
                if s.get("ok") is True and s.get("turn") is not None and (s.get("phase") is not None or s.get("phase_name") is not None):
                    break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError("Forge server not ready after 60s")

    # Preflight: fail fast if the env is misconfigured (assets, decks, endpoints, etc.)
    from rl_preflight import main as preflight_main
    t0 = time.time()
    rc = preflight_main()
    print(f"[train_ppo] preflight rc={rc} dt={time.time()-t0:.2f}s", flush=True)
    if rc != 0:
        raise SystemExit(rc)

    logdir = Path(__file__).resolve().parents[1] / "runs" / "ppo_v0"
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"[train_ppo] logdir={logdir}", flush=True)

    def mask_fn(e):
        return e.action_masks()

    # Training env: ActionMasker must be the outermost wrapper so MaskablePPO can see action_masks().
    # VecMonitor supports info_keywords directly (works with VecEnvs).
    info_keys = (
        "ep_dmg_to_opp",
        "ep_dmg_to_self",
        "ep_our_creatures_lost",
        "ep_opp_creatures_lost",
        "ep_spells_cast",
        "ep_activations_used",
        "ep_lands_played",
        "ep_commander_casts",
        "ep_commander_legal_steps",
        "ep_commander_in_actions_any",
        "ep_steps",
    )

    # Parallel envs (optional): set FORGE_N_ENVS to run multiple base URLs.
    # Example: FORGE_N_ENVS=4 FORGE_BASE_PORT=8799
    n_envs = int(os.environ.get("FORGE_N_ENVS", "1"))
    base_port = int(os.environ.get("FORGE_BASE_PORT", "8799"))

    def make_env(i: int):
        url = f"http://127.0.0.1:{base_port + i}"
        return lambda: ActionMasker(ForgeHttpEnv(base_url=url), mask_fn)

    if n_envs > 1:
        # fork tends to be more robust than forkserver/spawn for our HTTP-heavy env
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="fork")
    else:
        env = DummyVecEnv([make_env(0)])
    env = VecMonitor(env, info_keywords=info_keys)  # logs rollout + our episode metrics

    # Eval env (kept separate from training env). Use a single env on base_port.
    # NOTE: We use our custom eval callback for win_rate + domain metrics.
    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecMonitor(eval_env, info_keywords=info_keys)

    # In DEBUG_FAST mode, reduce rollout size so we get scalars quickly.
    n_steps = 128 if debug_fast else 2048
    batch_size = 64 if debug_fast else 256

    print(f"[train_ppo] building model (n_steps={n_steps}, batch_size={batch_size})", flush=True)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(logdir),
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.02,
    )

    # Eval cadence: Forge episodes are expensive, so avoid evaluating too frequently.
    eval_freq = 1_000 if debug_fast else 20_000

    # Use only our custom eval callback (win_rate + domain metrics). This avoids paying for two eval loops.
    extra_eval_cb = ForgeExtraEvalMetricsCallback(base_url=base, eval_freq=eval_freq, n_eval_episodes=10, record_replays="eval")

    bench_ts = os.environ.get("BENCH_TOTAL_TIMESTEPS")
    total_timesteps = int(bench_ts) if bench_ts else (20_000 if debug_fast else 1_000_000)
    print(f"[train_ppo] entering learn(total_timesteps={total_timesteps})", flush=True)
    model.learn(total_timesteps=total_timesteps, callback=[extra_eval_cb])
    print("[train_ppo] learn() returned; saving...", flush=True)
    model.save(str(logdir / "ppo_forge_v0"))
    print("[train_ppo] done", flush=True)


if __name__ == "__main__":
    main()
