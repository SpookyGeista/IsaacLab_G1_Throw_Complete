# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg, SceneEntityCfg
from isaaclab.envs import mdp
import torch

def ball_ground_contact(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"), threshold_z: float = 0.05):
    """Sofortige Termination: Ball berührt den Boden (keine Schonfrist)."""
    ball_pos = env.scene[asset_cfg.name].data.root_pos_w
    ball_height = ball_pos[:, 2]  # Z‑Koordinate
    return ball_height < threshold_z

def ball_at_rest_on_ground(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"), threshold_z: float = 0.20, vel_thresh: float = 0.05, rest_steps: int = 30):
    """Beende Episode, wenn der Ball am Boden liegt und zur Ruhe gekommen ist.

    - z < threshold_z
    - ||linear_velocity|| < vel_thresh für mindestens rest_steps
    """
    ball = env.scene[asset_cfg.name]
    z = ball.data.root_pos_w[:, 2]
    lin_vel = ball.data.root_lin_vel_w
    speed = torch.linalg.norm(lin_vel, dim=-1)

    # einfacher Zähler über episode_length_buf verwenden (konservativ)
    if hasattr(env, "episode_length_buf") and isinstance(env.episode_length_buf, torch.Tensor):
        steps = env.episode_length_buf
    else:
        steps = torch.full_like(z, fill_value=rest_steps + 1)

    near_ground = z < threshold_z
    low_speed = speed < vel_thresh
    long_enough = steps >= rest_steps
    return near_ground & low_speed & long_enough

def step_limit(env, max_steps: int = 500):
    """Termination: beendet Episode nach fester Schrittzahl (schrittbasiertes Timeout)."""
    if hasattr(env, "episode_length_buf") and isinstance(env.episode_length_buf, torch.Tensor):
        return env.episode_length_buf >= max_steps
    # wenn Buffer fehlt: nie auslösen
    return torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Schrittbasiertes Timeout
    time_out = TerminationTermCfg(
        func=step_limit, time_out=True
    )

    # Ball liegt zur Ruhe -> Episode beenden
    ball_rest = TerminationTermCfg(
        func=ball_at_rest_on_ground, time_out=False
    )

    # Sofort beenden, sobald Ball den Boden berührt
    ball_ground = TerminationTermCfg(
        func=ball_ground_contact, time_out=False
    )
