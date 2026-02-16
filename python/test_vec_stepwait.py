#!/usr/bin/env python3
import os
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from forge_gym_env import ForgeHttpEnv


def mask_fn(e):
    base = getattr(e, "unwrapped", e)
    return base.action_masks()


def main():
    base_port = int(os.environ.get("FORGE_BASE_PORT", "8800"))
    n_envs = int(os.environ.get("FORGE_N_ENVS", "4"))

    def make_env(i: int):
        url = f"http://127.0.0.1:{base_port + i}"
        return lambda: ActionMasker(ForgeHttpEnv(base_url=url), mask_fn)

    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    obs = env.reset()
    print("reset", obs.shape)
    for t in range(200):
        obs, rew, dones, infos = env.step([0] * n_envs)
        if t % 20 == 0:
            print("t", t, "rew", rew, "dones", dones)
    print("ok")


if __name__ == "__main__":
    main()
