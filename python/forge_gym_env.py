#!/usr/bin/env python3
"""Gymnasium env wrapper for the Forge RL v0 HTTP server (action_id-based).

Observation (Box):
  base(24) + hand vocab ids (HAND_N) + battlefield features (2 players * BF_N * 7)

Battlefield per slot encodes:
  [name_id, type_mask, tapped, sick, power, toughness, is_commander]

Notes:
- action masking is provided for sb3-contrib MaskablePPO via action_masks().
- invalid actions are converted to PASS but penalized.
- reward shaping includes life deltas + coarse creature-count deltas + light action bonuses.

Performance:
- cache legal actions returned by server responses
- use /advance_wait (long-poll) to reduce HTTP polling
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import requests

from card_vocab import CardVocab


def _json_post(sess: requests.Session, url: str, *, json_body: dict | None = None, timeout: float = 30.0, retries: int = 10):
    last_err = None
    import time
    for i in range(max(1, retries)):
        try:
            r = sess.post(url, json=json_body, timeout=timeout)
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.05 + 0.05 * i)
    raise last_err


def _json_get(sess: requests.Session, url: str, *, timeout: float = 10.0, retries: int = 10):
    last_err = None
    import time
    for i in range(max(1, retries)):
        try:
            r = sess.get(url, timeout=timeout)
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.05 + 0.05 * i)
    raise last_err


MAX_ACTIONS = 32
HAND_N = 10
BF_N = 12

# Reward shaping knobs
INVALID_ACTION_PENALTY = -0.1
DMG_TO_OPP_REWARD = 1.0       # + per life point removed from opponent
DMG_TO_SELF_PENALTY = 1.0     # - per life point lost

CAST_SPELL_REWARD = 0.0
CAST_COMMANDER_REWARD = 0.0
ACTIVATION_REWARD = 0.0
PLAY_LAND_REWARD = 0.0

# Tempo / anti-stall
STEP_TIME_PENALTY = 0.02
PASS_NOOP_PENALTY = 0.02  # penalty when we choose PASS despite other options

# Combat conversion
ATTACK_POWER_REWARD = 0.0        # per estimated attacking power when we choose an ATTACK_* action (non-NONE)
LETHAL_THREAT_BONUS = 0.0         # small bonus if we appear to have lethal on board (naive)

# Defense / blocks
BLOCK_ACTION_BONUS = 0.0          # small bonus for choosing a blocking macro (non-NONE)
PREVENT_DMG_REWARD = 0.0          # per point of (estimated incoming - actual dmg_to_self), clipped
PREVENT_DMG_CLIP = 20
BIG_HIT_THRESHOLD = 6
BIG_HIT_EXTRA_PENALTY = 0.0       # extra penalty per damage above threshold

# Board / threat value
CREATURE_LOSS_PENALTY = 0.01       # per creature lost
CREATURE_KILL_REWARD = 0.01        # per opponent creature lost
OPP_POWER_REMOVED_REWARD = 0.0   # per point of opponent battlefield power removed (naive)
OUR_POWER_GAIN_REWARD = 0.0     # per point of our battlefield power gained (naive)

BOARD_ADV_REWARD = 0.0           # tiny shaping: + per (our_creatures - opp_creatures) each step
BOARD_ADV_CLIP = 10

MAX_TURN = 25                      # truncate very long games to increase terminal feedback density
MAX_EP_STEPS = 400                 # hard cap on decision steps (prevents degenerate non-turn-advancing loops)
TERMINAL_WIN_BONUS = 100.0
TERMINAL_LOSS_PENALTY = 100.0

# commander zone ids (small categorical)
ZONE_TO_ID = {
    None: 0,
    "": 0,
    "Hand": 1,
    "Battlefield": 2,
    "Graveyard": 3,
    "Exile": 4,
    "Library": 5,
    "Command": 6,
}


class ForgeHttpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_url: str = "http://127.0.0.1:8799"):
        super().__init__()
        self.base = base_url.rstrip("/")
        self._sess = requests.Session()

        self.action_space = gym.spaces.Discrete(MAX_ACTIONS)

        # Observation vector:
        # base(24) + hand ids + battlefield features (2 players * BF_N * 7)
        obs_n = 24 + HAND_N + (2 * BF_N * 7)
        self.observation_space = gym.spaces.Box(low=-20000, high=20000, shape=(obs_n,), dtype=np.float32)

        self._last_state: dict[str, Any] | None = None
        self._last_actions: list[str] | None = None

        self._vocab = CardVocab("/home/ddroder/users/dan/forge_env/rl_v0/python/card_vocab.json")

        # Episode tracking for shaped rewards + diagnostics
        self._prev_p1_life: int | None = None
        self._prev_p2_life: int | None = None
        self._ep_dmg_to_opp: float = 0.0
        self._ep_dmg_to_self: float = 0.0

        self._prev_p1_creatures: int | None = None
        self._prev_p2_creatures: int | None = None
        self._ep_our_creatures_lost: float = 0.0
        self._ep_opp_creatures_lost: float = 0.0

        # Naive power sums (used for dense shaping)
        self._prev_p1_pow_sum: int | None = None
        self._prev_p2_pow_sum: int | None = None

        self._ep_spells_cast: float = 0.0
        self._ep_activations_used: float = 0.0
        self._ep_lands_played: float = 0.0
        self._ep_commander_casts: float = 0.0

        # Commander availability instrumentation
        self._ep_commander_legal_steps: float = 0.0  # in first MAX_ACTIONS
        self._ep_commander_in_actions_any: float = 0.0  # anywhere in action list
        self._ep_steps: float = 0.0

    # -----------------
    # HTTP + actions
    # -----------------

    def _actions_from_response(self, resp: dict[str, Any]) -> list[str] | None:
        if isinstance(resp.get("actions"), list):
            return list(resp.get("actions"))
        if isinstance(resp.get("legal_actions"), list):
            return list(resp.get("legal_actions"))
        return None

    def _get_actions(self) -> list[str]:
        if isinstance(self._last_actions, list) and self._last_actions:
            return list(self._last_actions)
        la = _json_get(self._sess, self.base + "/legal_actions", timeout=10, retries=10)
        acts = list(la.get("actions", ["PASS"]))
        self._last_actions = acts
        return acts

    def _mask_from_actions(self, actions: list[str]) -> np.ndarray:
        mask = np.zeros((MAX_ACTIONS,), dtype=np.int8)
        n = min(len(actions), MAX_ACTIONS)
        mask[:n] = 1
        mask[0] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        actions = self._get_actions()
        mask = self._mask_from_actions(actions)

        # Lightweight hard rules to avoid obvious blunders (lethal/passive play)
        try:
            s = self._last_state or {}
            p1_life = int(s.get("p1_life") or 40)
            p2_life = int(s.get("p2_life") or 40)

            # Decode battlefield (our side)
            def list_of(k: str, n: int, default):
                v = s.get(k)
                if not isinstance(v, list):
                    v = []
                v = list(v)[:n]
                if len(v) < n:
                    v = v + [default] * (n - len(v))
                return v

            p1_pow = [int(x or 0) for x in list_of("p1_bf_pow", BF_N, 0)]
            p1_tapped = [int(x or 0) for x in list_of("p1_bf_tapped", BF_N, 0)]
            p1_sick = [int(x or 0) for x in list_of("p1_bf_sick", BF_N, 0)]

            p2_names = [str(x or "") for x in list_of("p2_bf_names", BF_N, "")]
            p2_pow = [int(x or 0) for x in list_of("p2_bf_pow", BF_N, 0)]
            p2_tapped = [int(x or 0) for x in list_of("p2_bf_tapped", BF_N, 0)]

            # --- Attack-if-lethal ---
            has_attack_action = any(a.startswith("ATTACK_") for a in actions[:MAX_ACTIONS])
            if has_attack_action:
                # naive: sum power of creatures that can attack (untapped + not sick)
                our_attack_pow = sum(max(0, p) for p, t, sick in zip(p1_pow, p1_tapped, p1_sick) if t == 0 and sick == 0)
                if our_attack_pow >= p2_life and p2_life > 0:
                    # Don't allow doing nothing when we likely have lethal
                    for i, a in enumerate(actions[:MAX_ACTIONS]):
                        if a == "ATTACK_NONE":
                            mask[i] = 0
                    # Prefer ATTACK_ALL if present
                    for i, a in enumerate(actions[:MAX_ACTIONS]):
                        if a == "ATTACK_ALL":
                            mask[i] = 1

            # --- Block-if-lethal (approx) ---
            has_block_action = any(a.startswith("BLOCK_") for a in actions[:MAX_ACTIONS])
            if has_block_action:
                # naive: treat tapped opponent creatures with power>0 as likely attackers
                incoming_pow = sum(max(0, p) for p, t in zip(p2_pow, p2_tapped) if t == 1)
                if incoming_pow >= p1_life and p1_life > 0:
                    for i, a in enumerate(actions[:MAX_ACTIONS]):
                        if a == "BLOCK_NONE":
                            mask[i] = 0
                    # Ensure a defensive macro remains selectable
                    for pref in ("BLOCK_CHUMP_IF_LETHAL", "BLOCK_MAX", "BLOCK_TRADE"):
                        for i, a in enumerate(actions[:MAX_ACTIONS]):
                            if a.startswith(pref):
                                mask[i] = 1

            # --- Anti-bomb targeting bias ---
            # If opponent has a known "bomb" on board, discourage wasting activations targeting our own BF.
            bombs = ("Goreclaw, Terror of Qal Sisma", "Craterhoof Behemoth")
            bomb_present = any(any(b in n for b in bombs) for n in p2_names) or any(p >= 8 for p in p2_pow)
            if bomb_present:
                for i, a in enumerate(actions[:MAX_ACTIONS]):
                    if a.startswith("ACTIVATE:") and ":TGT_P1_BF:" in a:
                        # keep commander/self-synergy activations; only suppress obvious self-target removal loops
                        if any(x in a for x in ("Goblin Trashmaster", "Goblin Cratermaker")):
                            mask[i] = 0
        except Exception:
            pass

        return mask

    # -----------------
    # Observation
    # -----------------

    def _obs_from_state(self, s: dict[str, Any]) -> np.ndarray:
        def gi(k, default=0):
            v = s.get(k)
            return int(v) if isinstance(v, (int, float)) and v is not None else int(default)

        hand_names = s.get("p1_hand_names")
        if not isinstance(hand_names, list):
            hand_names = [""] * HAND_N
        hand_names = [str(x) if x is not None else "" for x in hand_names][:HAND_N]
        if len(hand_names) < HAND_N:
            hand_names = hand_names + [""] * (HAND_N - len(hand_names))
        hand_ids = [self._vocab.encode(n) for n in hand_names]

        zone = s.get("p1_commander_zone")
        zone_id = ZONE_TO_ID.get(zone, 0)

        def list_of(k: str, n: int, default):
            v = s.get(k)
            if not isinstance(v, list):
                v = []
            v = list(v)[:n]
            if len(v) < n:
                v = v + [default] * (n - len(v))
            return v

        p1_bf_names = [str(x or "") for x in list_of("p1_bf_names", BF_N, "")]
        p2_bf_names = [str(x or "") for x in list_of("p2_bf_names", BF_N, "")]
        p1_bf_name_ids = [self._vocab.encode(n) for n in p1_bf_names]
        p2_bf_name_ids = [self._vocab.encode(n) for n in p2_bf_names]

        p1_bf_types = [int(x or 0) for x in list_of("p1_bf_types", BF_N, 0)]
        p1_bf_tapped = [int(x or 0) for x in list_of("p1_bf_tapped", BF_N, 0)]
        p1_bf_sick = [int(x or 0) for x in list_of("p1_bf_sick", BF_N, 0)]
        p1_bf_pow = [int(x or 0) for x in list_of("p1_bf_pow", BF_N, 0)]
        p1_bf_tgh = [int(x or 0) for x in list_of("p1_bf_tgh", BF_N, 0)]
        p1_bf_is_cmd = [int(x or 0) for x in list_of("p1_bf_is_cmd", BF_N, 0)]

        p2_bf_types = [int(x or 0) for x in list_of("p2_bf_types", BF_N, 0)]
        p2_bf_tapped = [int(x or 0) for x in list_of("p2_bf_tapped", BF_N, 0)]
        p2_bf_sick = [int(x or 0) for x in list_of("p2_bf_sick", BF_N, 0)]
        p2_bf_pow = [int(x or 0) for x in list_of("p2_bf_pow", BF_N, 0)]
        p2_bf_tgh = [int(x or 0) for x in list_of("p2_bf_tgh", BF_N, 0)]
        p2_bf_is_cmd = [int(x or 0) for x in list_of("p2_bf_is_cmd", BF_N, 0)]

        bf_flat: list[int] = []
        for i in range(BF_N):
            bf_flat.extend([
                p1_bf_name_ids[i],
                p1_bf_types[i],
                p1_bf_tapped[i],
                p1_bf_sick[i],
                p1_bf_pow[i],
                p1_bf_tgh[i],
                p1_bf_is_cmd[i],
            ])
        for i in range(BF_N):
            bf_flat.extend([
                p2_bf_name_ids[i],
                p2_bf_types[i],
                p2_bf_tapped[i],
                p2_bf_sick[i],
                p2_bf_pow[i],
                p2_bf_tgh[i],
                p2_bf_is_cmd[i],
            ])

        return np.array(
            [
                gi("turn"), gi("phase_id"),
                gi("p1_life", 40), gi("p2_life", 40),
                gi("p1_hand"), gi("p2_hand"),
                gi("p1_creatures"), gi("p2_creatures"),
                gi("p1_lands"), gi("p2_lands"),
                gi("p1_untapped_lands"), gi("p2_untapped_lands"),
                gi("p1_mana_r"), gi("p1_mana_g"), gi("p1_mana_u"), gi("p1_mana_b"), gi("p1_mana_w"),
                gi("p2_mana_r"), gi("p2_mana_g"), gi("p2_mana_u"), gi("p2_mana_b"), gi("p2_mana_w"),
                gi("p1_commander_cast"), zone_id,
                *hand_ids,
                *bf_flat,
            ],
            dtype=np.float32,
        )

    # -----------------
    # Gym API
    # -----------------

    def reset(self, *, seed=None, options=None):
        _json_post(self._sess, self.base + "/reset", timeout=10, retries=10)
        ready = _json_get(self._sess, self.base + "/wait_ready", timeout=30, retries=10)
        self._last_state = ready
        self._last_actions = self._actions_from_response(ready)

        self._prev_p1_life = int(ready.get("p1_life") or 40)
        self._prev_p2_life = int(ready.get("p2_life") or 40)
        self._ep_dmg_to_opp = 0.0
        self._ep_dmg_to_self = 0.0

        self._prev_p1_creatures = int(ready.get("p1_creatures") or 0)
        self._prev_p2_creatures = int(ready.get("p2_creatures") or 0)
        self._ep_our_creatures_lost = 0.0
        self._ep_opp_creatures_lost = 0.0

        self._ep_spells_cast = 0.0
        self._ep_activations_used = 0.0
        self._ep_lands_played = 0.0
        self._ep_commander_casts = 0.0
        self._ep_commander_legal_steps = 0.0
        self._ep_commander_in_actions_any = 0.0
        self._ep_steps = 0.0

        # Reset power baselines
        try:
            def list_of(k: str, n: int, default):
                v = ready.get(k)
                if not isinstance(v, list):
                    v = []
                v = list(v)[:n]
                if len(v) < n:
                    v = v + [default] * (n - len(v))
                return v

            p1_pow = [int(x or 0) for x in list_of("p1_bf_pow", BF_N, 0)]
            p2_pow = [int(x or 0) for x in list_of("p2_bf_pow", BF_N, 0)]
            self._prev_p1_pow_sum = int(sum(max(0, p) for p in p1_pow))
            self._prev_p2_pow_sum = int(sum(max(0, p) for p in p2_pow))
        except Exception:
            self._prev_p1_pow_sum = None
            self._prev_p2_pow_sum = None

        actions = self._get_actions()
        obs = self._obs_from_state(ready)
        info = {
            "action_mask": self._mask_from_actions(actions),
            "actions": actions[:MAX_ACTIONS],
            "p1_life": float(self._prev_p1_life),
            "p2_life": float(self._prev_p2_life),
            "dmg_to_opp": 0.0,
            "dmg_to_self": 0.0,
            "ep_dmg_to_opp": 0.0,
            "ep_dmg_to_self": 0.0,
            "ep_our_creatures_lost": 0.0,
            "ep_opp_creatures_lost": 0.0,
            "ep_spells_cast": 0.0,
            "ep_activations_used": 0.0,
            "ep_lands_played": 0.0,
            "ep_commander_casts": 0.0,
            "ep_commander_legal_steps": 0.0,
            "ep_commander_in_actions_any": 0.0,
        }
        return obs, info

    def step(self, action: int):
        assert self._last_state is not None

        actions = self._get_actions()
        invalid = False
        if action < 0 or action >= MAX_ACTIONS:
            invalid = True
            action = 0
        if action >= min(len(actions), MAX_ACTIONS):
            invalid = True
            action = 0

        # Pre-step naive estimates (used for reward shaping)
        pre_incoming_pow = 0
        pre_attack_pow = 0
        pre_has_lethal = False

        # Hard overrides for obvious tactical blunders (aligns with action_masks heuristics)
        try:
            chosen = actions[action] if action < min(len(actions), MAX_ACTIONS) else "PASS"
            s0 = self._last_state or {}
            p1_life = int(s0.get("p1_life") or 40)
            p2_life = int(s0.get("p2_life") or 40)

            def list_of(k: str, n: int, default):
                v = s0.get(k)
                if not isinstance(v, list):
                    v = []
                v = list(v)[:n]
                if len(v) < n:
                    v = v + [default] * (n - len(v))
                return v

            p1_pow = [int(x or 0) for x in list_of("p1_bf_pow", BF_N, 0)]
            p1_tapped = [int(x or 0) for x in list_of("p1_bf_tapped", BF_N, 0)]
            p1_sick = [int(x or 0) for x in list_of("p1_bf_sick", BF_N, 0)]
            p2_pow = [int(x or 0) for x in list_of("p2_bf_pow", BF_N, 0)]
            p2_tapped = [int(x or 0) for x in list_of("p2_bf_tapped", BF_N, 0)]

            pre_attack_pow = int(sum(max(0, p) for p, t, sick in zip(p1_pow, p1_tapped, p1_sick) if t == 0 and sick == 0))
            pre_incoming_pow = int(sum(max(0, p) for p, t in zip(p2_pow, p2_tapped) if t == 1))
            pre_has_lethal = bool(p2_life > 0 and pre_attack_pow >= p2_life)

            if chosen == "ATTACK_NONE":
                if pre_has_lethal:
                    # override to ATTACK_ALL if available
                    if "ATTACK_ALL" in actions[:MAX_ACTIONS]:
                        action = actions[:MAX_ACTIONS].index("ATTACK_ALL")

            if chosen == "BLOCK_NONE":
                if p1_life > 0 and pre_incoming_pow >= p1_life:
                    for pref in ("BLOCK_CHUMP_IF_LETHAL", "BLOCK_MAX", "BLOCK_TRADE"):
                        for i, a in enumerate(actions[:MAX_ACTIONS]):
                            if a.startswith(pref):
                                action = i
                                break
                        else:
                            continue
                        break
        except Exception:
            pass

        # Single-call stepping (server handles: advance-to-wait, apply action, advance-to-next-wait)
        try:
            s = None
            for _ in range(5):
                s = _json_post(self._sess, self.base + "/step_wait", json_body={"action_id": int(action)}, timeout=30, retries=10)
                if isinstance(s, dict) and s.get("ok") is True and s.get("turn") is not None and (s.get("phase") is not None or s.get("phase_name") is not None):
                    break
                # If server says no_game or isn't ready yet, try to reset and retry.
                try:
                    _json_post(self._sess, self.base + "/reset", timeout=15, retries=5)
                    _json_get(self._sess, self.base + "/wait_ready", timeout=30, retries=5)
                except Exception:
                    pass
            if not (isinstance(s, dict) and s.get("ok") is True):
                raise RuntimeError(f"step_wait failed: {s}")
        except Exception as e:
            # IMPORTANT: never raise from step() inside SubprocVecEnv worker, or it kills the worker (BrokenPipe/EOF).
            # Attempt a reset; if that fails, return a safe terminal transition.
            try:
                _json_post(self._sess, self.base + "/reset", timeout=15, retries=5)
                ready = _json_get(self._sess, self.base + "/wait_ready", timeout=30, retries=5)
                self._last_state = ready
                self._last_actions = self._actions_from_response(ready)
                obs = self._obs_from_state(ready)
                # Always include VecMonitor info_keywords to avoid KeyError
                info = {
                    "winner": None,
                    "waiting": ready.get("waiting"),
                    "action_taken": "",
                    "actions": (self._last_actions or [])[:MAX_ACTIONS],
                    "action_mask": self._mask_from_actions(self._last_actions or ["PASS"]),
                    "invalid_action": True,
                    "error": str(e),
                    "ep_dmg_to_opp": float(self._ep_dmg_to_opp),
                    "ep_dmg_to_self": float(self._ep_dmg_to_self),
                    "ep_our_creatures_lost": float(self._ep_our_creatures_lost),
                    "ep_opp_creatures_lost": float(self._ep_opp_creatures_lost),
                    "ep_spells_cast": float(self._ep_spells_cast),
                    "ep_activations_used": float(self._ep_activations_used),
                    "ep_lands_played": float(self._ep_lands_played),
                    "ep_commander_casts": float(self._ep_commander_casts),
                    "ep_commander_legal_steps": float(self._ep_commander_legal_steps),
                    "ep_commander_in_actions_any": float(self._ep_commander_in_actions_any),
                    "ep_steps": float(self._ep_steps),
                }
                return obs, -1.0, True, True, info
            except Exception:
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                info = {
                    "winner": None,
                    "waiting": None,
                    "action_taken": "",
                    "invalid_action": True,
                    "error": str(e),
                    "actions": ["PASS"],
                    "action_mask": self._mask_from_actions(["PASS"]),
                    "ep_dmg_to_opp": float(self._ep_dmg_to_opp),
                    "ep_dmg_to_self": float(self._ep_dmg_to_self),
                    "ep_our_creatures_lost": float(self._ep_our_creatures_lost),
                    "ep_opp_creatures_lost": float(self._ep_opp_creatures_lost),
                    "ep_spells_cast": float(self._ep_spells_cast),
                    "ep_activations_used": float(self._ep_activations_used),
                    "ep_lands_played": float(self._ep_lands_played),
                    "ep_commander_casts": float(self._ep_commander_casts),
                    "ep_commander_legal_steps": float(self._ep_commander_legal_steps),
                    "ep_commander_in_actions_any": float(self._ep_commander_in_actions_any),
                    "ep_steps": float(self._ep_steps),
                }
                return obs, -1.0, True, True, info

        # Update cached actions from server response
        acts = self._actions_from_response(s)
        if acts is not None:
            self._last_actions = acts

        # Commander availability at decision point (based on returned action list)
        try:
            acts0 = self._get_actions()
            if any(a == "CAST_COMMANDER" for a in acts0):
                self._ep_commander_in_actions_any += 1.0
            if any(a == "CAST_COMMANDER" for a in acts0[:MAX_ACTIONS]):
                self._ep_commander_legal_steps += 1.0
        except Exception:
            pass

        self._last_state = s

        obs = self._obs_from_state(s)
        reward = 0.0  # override server shaping; reward computed below

        act_taken = str(s.get("action_taken") or "")

        # Penalize PASS/no-op if other actions exist (anti-stall / encourage progress)
        try:
            if act_taken == "PASS":
                acts0 = self._get_actions()
                if any(a != "PASS" for a in (acts0 or [])[:MAX_ACTIONS]):
                    reward -= PASS_NOOP_PENALTY
        except Exception:
            pass
        if act_taken.startswith("CAST_COMMANDER"):
            self._ep_commander_casts += 1.0
            reward += CAST_COMMANDER_REWARD
        elif act_taken.startswith("CAST:"):
            self._ep_spells_cast += 1.0
            reward += CAST_SPELL_REWARD
        elif act_taken.startswith("ACTIVATE:"):
            self._ep_activations_used += 1.0
            reward += ACTIVATION_REWARD
        elif act_taken.startswith("PLAY_LAND:"):
            self._ep_lands_played += 1.0
            reward += PLAY_LAND_REWARD

        def _int_default(v, default: int):
            return default if v is None else int(v)

        p1_life = _int_default(s.get("p1_life"), 40)
        p2_life = _int_default(s.get("p2_life"), 40)
        prev_p1 = self._prev_p1_life if self._prev_p1_life is not None else p1_life
        prev_p2 = self._prev_p2_life if self._prev_p2_life is not None else p2_life

        dmg_to_opp = max(0, prev_p2 - p2_life)
        dmg_to_self = max(0, prev_p1 - p1_life)

        self._ep_dmg_to_opp += float(dmg_to_opp)
        self._ep_dmg_to_self += float(dmg_to_self)

        reward += DMG_TO_OPP_REWARD * float(dmg_to_opp)
        reward -= DMG_TO_SELF_PENALTY * float(dmg_to_self)

        # Extra penalty for big chunks of combat damage (often preventable)
        if dmg_to_self > BIG_HIT_THRESHOLD:
            reward -= BIG_HIT_EXTRA_PENALTY * float(dmg_to_self - BIG_HIT_THRESHOLD)

        # Defense shaping: reward preventing incoming damage (naive estimate from pre-step)
        try:
            prevented = max(0, int(pre_incoming_pow) - int(dmg_to_self))
            prevented = min(PREVENT_DMG_CLIP, prevented)
            reward += PREVENT_DMG_REWARD * float(prevented)
        except Exception:
            pass

        self._prev_p1_life = p1_life
        self._prev_p2_life = p2_life

        p1_creat = int(s.get("p1_creatures") or 0)
        p2_creat = int(s.get("p2_creatures") or 0)
        prev_c1 = self._prev_p1_creatures if self._prev_p1_creatures is not None else p1_creat
        prev_c2 = self._prev_p2_creatures if self._prev_p2_creatures is not None else p2_creat

        our_lost = max(0, prev_c1 - p1_creat)
        opp_lost = max(0, prev_c2 - p2_creat)
        self._ep_our_creatures_lost += float(our_lost)
        self._ep_opp_creatures_lost += float(opp_lost)

        reward -= CREATURE_LOSS_PENALTY * float(our_lost)
        reward += CREATURE_KILL_REWARD * float(opp_lost)

        # small dense shaping for board advantage (kept tiny to avoid reward hacking)
        adv = max(-BOARD_ADV_CLIP, min(BOARD_ADV_CLIP, p1_creat - p2_creat))
        reward += BOARD_ADV_REWARD * float(adv)

        # Dense shaping using naive battlefield power sums (helps distinguish "20 goblins" vs "2 goblins")
        try:
            def _sum_pow(prefix: str) -> int:
                arr = s.get(prefix + "_bf_pow")
                if not isinstance(arr, list):
                    return 0
                return int(sum(max(0, int(x or 0)) for x in arr[:BF_N]))

            p1_pow_sum = _sum_pow("p1")
            p2_pow_sum = _sum_pow("p2")
            prev_p1_pow_sum = self._prev_p1_pow_sum if self._prev_p1_pow_sum is not None else p1_pow_sum
            prev_p2_pow_sum = self._prev_p2_pow_sum if self._prev_p2_pow_sum is not None else p2_pow_sum

            # Reward removing opponent power (threats) and gaining our own power
            opp_pow_removed = max(0, prev_p2_pow_sum - p2_pow_sum)
            our_pow_gained = max(0, p1_pow_sum - prev_p1_pow_sum)
            reward += OPP_POWER_REMOVED_REWARD * float(opp_pow_removed)
            reward += OUR_POWER_GAIN_REWARD * float(our_pow_gained)

            self._prev_p1_pow_sum = p1_pow_sum
            self._prev_p2_pow_sum = p2_pow_sum
        except Exception:
            pass

        self._prev_p1_creatures = p1_creat
        self._prev_p2_creatures = p2_creat

        # Tempo: tiny per-step cost to discourage stalling
        reward -= STEP_TIME_PENALTY

        # Combat conversion shaping
        try:
            if act_taken.startswith("ATTACK_") and act_taken != "ATTACK_NONE":
                reward += ATTACK_POWER_REWARD * float(pre_attack_pow)
            if pre_has_lethal:
                reward += LETHAL_THREAT_BONUS
            if act_taken.startswith("BLOCK_") and act_taken != "BLOCK_NONE":
                reward += BLOCK_ACTION_BONUS
        except Exception:
            pass

        if invalid:
            reward += INVALID_ACTION_PENALTY

        terminated = bool(s.get("done"))
        truncated = False

        # Terminal shaping: make wins/losses louder than incremental shaping
        if terminated:
            w = s.get("winner")
            if isinstance(w, str) and "External" in w:
                reward += TERMINAL_WIN_BONUS
            elif isinstance(w, str) and w:
                reward -= TERMINAL_LOSS_PENALTY
        # Truncate very long episodes to improve learning signal density
        try:
            if int(s.get("turn") or 0) >= MAX_TURN:
                truncated = True
        except Exception:
            pass

        # Also cap on raw decision steps (turn can stall if we're stuck in priority loops)
        try:
            if self._ep_steps >= float(MAX_EP_STEPS):
                truncated = True
        except Exception:
            pass

        next_actions = self._get_actions()
        self._ep_steps += 1.0
        info = {
            "winner": s.get("winner"),
            "waiting": s.get("waiting"),
            "action_taken": act_taken,
            "actions": next_actions[:MAX_ACTIONS],
            "action_mask": self._mask_from_actions(next_actions),
            "invalid_action": invalid,

            "p1_life": float(p1_life),
            "p2_life": float(p2_life),
            "dmg_to_opp": float(dmg_to_opp),
            "dmg_to_self": float(dmg_to_self),

            # Episode metrics (VecMonitor info_keywords expects these keys when done)
            "ep_dmg_to_opp": float(self._ep_dmg_to_opp),
            "ep_dmg_to_self": float(self._ep_dmg_to_self),
            "ep_our_creatures_lost": float(self._ep_our_creatures_lost),
            "ep_opp_creatures_lost": float(self._ep_opp_creatures_lost),
            "ep_spells_cast": float(self._ep_spells_cast),
            "ep_activations_used": float(self._ep_activations_used),
            "ep_lands_played": float(self._ep_lands_played),
            "ep_commander_casts": float(self._ep_commander_casts),
            "ep_commander_legal_steps": float(self._ep_commander_legal_steps),
            "ep_commander_in_actions_any": float(self._ep_commander_in_actions_any),
            "ep_steps": float(self._ep_steps),
        }
        # Defensive: ensure keys exist even if something upstream returns a partial dict
        for k in (
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
        ):
            info.setdefault(k, 0.0)

        return obs, reward, terminated, truncated, info
