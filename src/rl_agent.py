"""
rl_agent.py
PPO-based reinforcement learning agent for coordinated 3-intersection
traffic signal control using stable-baselines3 + SUMO/TraCI.

Observation space (per junction × 3):
  - queue_ew: queue length on E-W approaches (normalised 0-1)
  - queue_ns: queue length on N-S approaches (normalised 0-1)
  - phase:    current phase (0=EW green, 1=NS green, one-hot encoded)
  → Total obs dim = 3 * 4 = 12

Action space: Discrete(8)
  Each bit in [0..7] controls one junction:
    bit 0 → TL0 request_switch, bit 1 → TL1 request_switch, bit 2 → TL2 request_switch
  (1 = request phase switch at this step; 0 = keep current phase)

Reward: negative total delay across all 3 junctions + even-flow bonus
"""

import os
import subprocess
import sys
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ── Optional stable-baselines3 import (only needed for RL training) ──────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ── TraCI import ──────────────────────────────────────────────────────────────
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
SUMO_CFG      = os.path.join(os.path.dirname(__file__), "..", "sumo", "config.sumocfg")
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_jaam_ctrl")
SIM_DURATION  = 1800          # seconds
MAX_QUEUE_NORM = 20.0          # vehicles (for normalisation)
CONTROL_STEP  = 10             # agent acts every N simulation steps
MIN_PHASE_DUR = 15             # minimum seconds before a switch is honoured
YELLOW_DUR    = 5

TL_IDS = ["TL0", "TL1", "TL2"]

JUNCTION_EDGES = {
    "TL0": {"ew": ["W0J0", "J1J0"], "ns": ["N0J0", "S0J0"]},
    "TL1": {"ew": ["J0J1", "J2J1"], "ns": ["N1J1", "S1J1"]},
    "TL2": {"ew": ["J1J2"],          "ns": ["N2J2", "S2J2"]},
}


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium Environment
# ─────────────────────────────────────────────────────────────────────────────

class CorridorEnv(gym.Env):
    """
    SUMO-based Gymnasium environment for 3-intersection PPO training.
    Each episode = 1 full simulation (SIM_DURATION steps).
    """

    metadata = {"render_modes": []}

    def __init__(self, sumo_binary: str = "sumo", seed: int = 42):
        super().__init__()
        self.sumo_binary = sumo_binary
        self._seed = seed

        # 12-dim continuous observation
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        # 8 discrete joint actions (3 binary bits)
        self.action_space = spaces.Discrete(8)

        self._step_count   = 0
        self._phase_timer  = {tl: 0 for tl in TL_IDS}   # seconds in current phase
        self._current_phase = {tl: 0 for tl in TL_IDS}  # 0=EW, 1=NS
        self._traci_open   = False

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._traci_open:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_open = False

        sumo_cmd = [
            self.sumo_binary,
            "-c", SUMO_CFG,
            "--seed", str(self._seed),
            "--no-warnings",
            "--no-step-log",
            "--quit-on-end",
        ]
        traci.start(sumo_cmd)
        self._traci_open = True
        self._step_count = 0
        self._phase_timer  = {tl: 0 for tl in TL_IDS}
        self._current_phase = {tl: 0 for tl in TL_IDS}

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert self._traci_open, "Call reset() before step()"

        # Decode action bits
        switches = {
            "TL0": bool(action & 1),
            "TL1": bool(action & 2),
            "TL2": bool(action & 4),
        }

        total_delay_before = self._total_delay()

        # Run CONTROL_STEP simulation steps
        for _ in range(CONTROL_STEP):
            if traci.simulation.getTime() >= SIM_DURATION:
                break
            for tl in TL_IDS:
                self._phase_timer[tl] += 1
                # Apply switch request if minimum phase duration met
                if switches[tl] and self._phase_timer[tl] >= MIN_PHASE_DUR:
                    self._switch_phase(tl)
            traci.simulationStep()
            self._step_count += 1

        total_delay_after = self._total_delay()
        reward = self._compute_reward(total_delay_before, total_delay_after)

        done = traci.simulation.getTime() >= SIM_DURATION
        if done:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_open = False

        obs = self._get_obs() if not done else np.zeros(12, dtype=np.float32)
        return obs, reward, done, False, {}

    def close(self):
        if self._traci_open:
            try:
                traci.close()
            except Exception:
                pass
            self._traci_open = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        obs = []
        for tl in TL_IDS:
            edges = JUNCTION_EDGES[tl]
            q_ew = min(self._sum_queue(edges["ew"]) / MAX_QUEUE_NORM, 1.0)
            q_ns = min(self._sum_queue(edges["ns"]) / MAX_QUEUE_NORM, 1.0)
            phase = self._current_phase[tl]
            obs.extend([q_ew, q_ns, float(phase == 0), float(phase == 1)])
        return np.array(obs, dtype=np.float32)

    def _sum_queue(self, edges: list) -> float:
        total = 0.0
        for e in edges:
            try:
                vids = traci.edge.getLastStepVehicleIDs(e)
                total += sum(1 for v in vids if traci.vehicle.getSpeed(v) < 0.1)
            except Exception:
                pass
        return total

    def _total_delay(self) -> float:
        """Sum of waiting time across all vehicles currently in simulation."""
        try:
            return sum(
                traci.vehicle.getWaitingTime(v)
                for v in traci.vehicle.getIDList()
            )
        except Exception:
            return 0.0

    def _compute_reward(self, delay_before: float, delay_after: float) -> float:
        """
        Reward = reduction in delay (positive when delay drops).
        Penalise uneven flow across junctions.
        """
        delay_reduction = (delay_before - delay_after) / max(1.0, delay_before)

        # Even-flow bonus: penalise if one junction is much worse than others
        queues = []
        for tl in TL_IDS:
            edges = JUNCTION_EDGES[tl]
            q = self._sum_queue(edges["ew"]) + self._sum_queue(edges["ns"])
            queues.append(q)
        flow_penalty = np.std(queues) / max(1.0, np.mean(queues))

        return float(delay_reduction - 0.1 * flow_penalty)

    def _switch_phase(self, tl: str):
        """Toggle between EW-green and NS-green with yellow transition."""
        try:
            current = self._current_phase[tl]
            if current == 0:
                # EW-green → EW-yellow → NS-green
                traci.trafficlight.setPhase(tl, 1)   # yellow
                traci.trafficlight.setPhaseDuration(tl, YELLOW_DUR)
                self._current_phase[tl] = 1
            else:
                traci.trafficlight.setPhase(tl, 3)   # NS yellow
                traci.trafficlight.setPhaseDuration(tl, YELLOW_DUR)
                self._current_phase[tl] = 0
            self._phase_timer[tl] = 0
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Training function (called from Streamlit on button click)
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(total_timesteps: int = 3000, progress_callback=None) -> str:
    """
    Train a PPO agent and save to MODEL_PATH.
    Returns MODEL_PATH on success, raises on failure.

    progress_callback(step, total) can be used to update a Streamlit progress bar.
    """
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is not installed. "
            "Run: pip install stable-baselines3"
        )
    if not TRACI_AVAILABLE:
        raise ImportError("traci is not installed. Install SUMO and its Python bindings.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    env = CorridorEnv(sumo_binary="sumo")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        gamma=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        tensorboard_log=None,
    )

    class _ProgressCB:
        def __init__(self, total, cb):
            self.total = total
            self.cb    = cb
            self.calls = 0

        def __call__(self, locals_, globals_):
            self.calls += 1
            if self.cb:
                self.cb(min(self.calls * 256, self.total), self.total)
            return True

    cb = _ProgressCB(total_timesteps, progress_callback)
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(MODEL_PATH)
    env.close()
    return MODEL_PATH


def load_ppo_model():
    """Load a previously trained model. Returns None if not found."""
    if not SB3_AVAILABLE:
        return None
    path = MODEL_PATH + ".zip"
    if not os.path.exists(path):
        return None
    return PPO.load(MODEL_PATH)


def run_ppo_episode(model) -> dict:
    """
    Run one full simulation episode with a trained PPO model.
    Returns metrics dict.
    """
    env = CorridorEnv(sumo_binary="sumo")
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action))
        total_reward += reward
        steps += 1
    env.close()
    return {"total_reward": total_reward, "steps": steps}
