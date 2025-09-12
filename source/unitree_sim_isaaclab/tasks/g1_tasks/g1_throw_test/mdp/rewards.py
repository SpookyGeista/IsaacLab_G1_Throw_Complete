import torch

def reward_ball_height(env, scale: float = 1.0):
    """Einfacher Reward: Höhe des Balls (Z) belohnen.

    Args:
        env: Manager‑Env mit Szene.
        scale: Multiplikator für die Belohnung.

    Returns:
        Tensor [num_envs] mit Rewardwerten.
    """
    ball = env.scene["ball"]
    z = ball.data.root_pos_w[:, 2]
    return scale * z


import torch

def reward_ball_xy_distance_gauss(env, target_xy_dist: float = 1.0, sigma: float = 0.25, scale: float = 1.0):
    # 1) Hole die XY-Position des Balls (Form: [num_envs, 2]).
    ball_xy = env.scene["ball"].data.root_pos_w[:, :2]

    # 2) Robust: Lese den im Reset gesetzten Referenzpunkt; falls noch nicht vorhanden -> None.
    spawn_xy = getattr(env, "_ball_spawn_xy", None)

    # 3) Fallback: Wenn Referenz nicht gesetzt (z. B. erster Step), gib neutralen 0‑Reward zurück.
    if spawn_xy is None:
        return torch.zeros((env.num_envs,), device=env.device)

    # 4) Sicherheitsprüfungen: gleiche Form und auf demselben Device (nur in Debug sinnvoll).
    # assert spawn_xy.shape == ball_xy.shape and spawn_xy.device == ball_xy.device

    # 5) Radiale XY‑Distanz zum gespeicherten Spawnpunkt.
    d_xy = torch.linalg.norm(ball_xy - spawn_xy, dim=-1)

    # 6) Abweichung vom Zielradius.
    diff = d_xy - target_xy_dist

    # 7) Gauß‑Glocke: r = exp(-0.5 * (diff/sigma)^2).
    r = torch.exp(-0.5 * (diff / sigma) ** 2)

    # 8) Skalierung (Gewichtung) der Belohnung.
    return scale * r