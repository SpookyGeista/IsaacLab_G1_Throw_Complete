# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Events for G1 Throwing Test Task"""

import torch


def _print(msg: str):
    # lightweight print tag for easy grepping
    print(f"[ResetHand] {msg}")


def reset_hand_position(env, **kwargs):
    """Reset event to set hand position for throwing - rotate right hand palm 90 degrees."""
    
    robot = env.scene["robot"]
    device = env.device

    # dump basic info
    dof_names = robot.dof_names
    _print(f"Robot has {len(dof_names)} DOFs")
    # show first few names for quick check
    _print(f"First DOFs: {dof_names[:10]}")

    target_angle = 1.5708  # 90 deg

    # helper to set a single joint and log
    def set_joint_by_name(joint_name: str, angle: float):
        try:
            idx = robot.get_dof_index(joint_name)
        except Exception as e:
            _print(f"Error get_dof_index('{joint_name}'): {e}")
            return False
        if idx is None or (isinstance(idx, int) and idx < 0):
            _print(f"Joint '{joint_name}' not found")
            return False
        # read current
        try:
            current = robot.get_joint_positions(indices=torch.tensor([idx], device=device))
            _print(f"{joint_name} (idx {idx}) current: {current.cpu().numpy().tolist()}")
        except Exception as e:
            _print(f"Could not read current for '{joint_name}': {e}")
        # set new
        try:
            robot.set_joint_positions(
                positions=torch.tensor([angle], device=device),
                indices=torch.tensor([idx], device=device),
            )
            _print(f"Set {joint_name} (idx {idx}) -> {angle} rad")
            return True
        except Exception as e:
            _print(f"Failed setting '{joint_name}': {e}")
            return False

    # set all three wrist joints to target and then read back current values
    names = ["right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint"]
    any_set = False
    for name in names:
        if set_joint_by_name(name, target_angle):
            any_set = True

    # read back all three for verification
    try:
        dof_indices = []
        for name in names:
            try:
                idx = robot.get_dof_index(name)
                dof_indices.append(idx)
            except Exception:
                dof_indices.append(None)
        valid_indices = [i for i in dof_indices if isinstance(i, int) and i >= 0]
        if valid_indices:
            vals = robot.get_joint_positions(indices=torch.tensor(valid_indices, device=device))
            _print(f"Readback wrist joints indices={valid_indices} values={vals.cpu().numpy().tolist()}")
        else:
            _print("No valid wrist indices to read back.")
    except Exception as e:
        _print(f"Readback failed: {e}")

    if not any_set:
        _print("No wrist joint updated. Check joint naming against USD.")
    else:
        _print("Wrist joints set to target (attempted).")


def reset_ball_and_robot(env, env_ids=None, **kwargs):
    """Reset event to reset ball and robot to initial positions (batched).

    Isaac Lab ruft Events mit optionalen `env_ids` auf. Diese Implementierung
    setzt Roboter- und Ballzustände nur für die angegebenen Envs oder für alle.
    """
    
    _print("Resetting ball and robot to initial positions...")

    # Guard: ersten Reset nach Stage-Ladung überspringen, um GPU-Semaphore-Issues zu vermeiden
    if not hasattr(env, "_first_reset_done"):
        env._first_reset_done = True
        _print("Skip first reset write to sim (guard)")
        return
    
    # Reset robot to initial joint positions
    robot = env.scene["robot"]
    device = env.device
    num_envs = getattr(env, "num_envs", 1)

    # Ziel-Env-IDs bestimmen
    if env_ids is None:
        if num_envs == 1:
            env_ids_tensor = None  # Single-env: ohne env_ids schreiben (stabiler)
        else:
            env_ids_tensor = torch.arange(num_envs, device=device, dtype=torch.long)
    else:
        env_ids_tensor = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    
    # Get initial joint positions from the scene configuration (tensor-only writes)
    initial_joint_pos = robot.data.default_joint_pos.clone()
    if env_ids_tensor is None:
        # single-env
        if initial_joint_pos.ndim == 1:
            robot.data.joint_pos[...] = initial_joint_pos
        else:
            robot.data.joint_pos[...] = initial_joint_pos[0]
        if hasattr(robot.data, "joint_vel"):
            robot.data.joint_vel.zero_()
    else:
        # multi-env indexed
        if initial_joint_pos.ndim == 1:
            robot.data.joint_pos[env_ids_tensor, :] = initial_joint_pos
        else:
            robot.data.joint_pos[env_ids_tensor, :] = initial_joint_pos[env_ids_tensor, :]
        if hasattr(robot.data, "joint_vel"):
            robot.data.joint_vel[env_ids_tensor, :].zero_()
    
    # Reset ball to 2 cm under the link 'right_hand_index_0_link' (world Z down)
    ball = env.scene["ball"]
    link_name = "right_hand_thumb_2_link"
    body_names = robot.data.body_names
    try:
        link_idx = body_names.index(link_name)
    except ValueError:
        _print(f"Link '{link_name}' not found in body_names; skipping ball placement override")
        link_idx = None

    if link_idx is not None:
        # world pose of the link for all envs
        link_pos_w_all = robot.data.body_pos_w[:, link_idx, :]    # (N,3)
        link_quat_w_all = robot.data.body_quat_w[:, link_idx, :]  # (N,4) as (w,x,y,z)

        # local offset: 1 cm along link local -Z (runter, vorne,links)(-Z, Y, X)
        local_off = torch.tensor([0.04, 0.03, 0.00], device=device)

        def quat_apply_batched(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
            """Apply unit quaternion(s) to vector(s). quat as (N,4) (w,x,y,z), vec as (3,) or (N,3)."""
            if vec.ndim == 1:
                vec = vec.unsqueeze(0).expand(quat_wxyz.shape[0], 3)
            w, x, y, z = quat_wxyz.unbind(dim=-1)
            # q * v * q^{-1} with v as pure quaternion (0, v)
            # Optimized equivalent rotation matrix application
            ww = w*w; xx = x*x; yy = y*y; zz = z*z
            xy = x*y; xz = x*z; yz = y*z; wx = w*x; wy = w*y; wz = w*z
            rotm = torch.stack([
                torch.stack([ww+xx-yy-zz, 2*(xy-wz),       2*(xz+wy)], dim=-1),
                torch.stack([2*(xy+wz),     ww-xx+yy-zz,   2*(yz-wx)], dim=-1),
                torch.stack([2*(xz-wy),     2*(yz+wx),     ww-xx-yy+zz], dim=-1)
            ], dim=-2)  # (N,3,3)
            return (rotm @ vec.unsqueeze(-1)).squeeze(-1)

        if env_ids_tensor is None:
            world_off = quat_apply_batched(link_quat_w_all[0:1, :], local_off)[0]
            pos_w = link_pos_w_all[0] + world_off
            rot_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            ball.data.root_state_w[..., 0:3] = pos_w
            ball.data.root_state_w[..., 3:7] = rot_w
            ball.data.root_state_w[..., 7:13] = 0.0
        else:
            k = env_ids_tensor.numel()
            world_off = quat_apply_batched(link_quat_w_all[env_ids_tensor, :], local_off)  # (K,3)
            pos_w = link_pos_w_all[env_ids_tensor] + world_off
            rot_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(k, 4)
            ball.data.root_state_w[env_ids_tensor, 0:3] = pos_w
            ball.data.root_state_w[env_ids_tensor, 3:7] = rot_w
            ball.data.root_state_w[env_ids_tensor, 7:13] = 0.0
    else:
        # Fallback: do nothing (scene/default placement remains)
        pass
    
    _print("Ball and robot reset to initial positions!")

    # WICHTIG: Zustände zurück in die PhysX-Sim schreiben (nur im Reset-Kontext)
    try:
        # Robot joint state -> sim
        # Nutze set_* API für maximale Kompatibilität (nur im Reset-Kontext aufrufen)
        try:
            if hasattr(robot, "set_joint_positions"):
                robot.set_joint_positions(positions=robot.data.joint_pos.clone())
            if hasattr(robot, "set_joint_velocities") and hasattr(robot.data, "joint_vel"):
                robot.set_joint_velocities(velocities=robot.data.joint_vel.clone())
        except Exception as e:
            _print(f"set_joint_* failed (non-fatal): {e}")

        # Ball root pose/velocity -> sim
        if hasattr(ball, "write_root_pose_to_sim") and hasattr(ball, "write_root_velocity_to_sim"):
            # clone() gegen Aliasing und in erwartete Form bringen
            pos_w = ball.data.root_state_w[..., 0:3].clone()
            rot_w = ball.data.root_state_w[..., 3:7].clone()
            lin_w = ball.data.root_state_w[..., 7:10].clone()
            ang_w = ball.data.root_state_w[..., 10:13].clone()
            if pos_w.ndim == 1:
                pos_w = pos_w.unsqueeze(0)
                rot_w = rot_w.unsqueeze(0)
                lin_w = lin_w.unsqueeze(0)
                ang_w = ang_w.unsqueeze(0)
            env_ids_arg = None if env_ids_tensor is None else env_ids_tensor
            pose_w = torch.cat([pos_w, rot_w], dim=-1)  # (N,7)
            vel_w = torch.cat([lin_w, ang_w], dim=-1)  # (N,6)
            # aktuelle API: write_root_pose_to_sim(root_pose_w, env_ids=None)
            ball.write_root_pose_to_sim(pose_w, env_ids_arg)
            # aktuelle API: write_root_velocity_to_sim(root_vel_w, env_ids=None)
            ball.write_root_velocity_to_sim(vel_w, env_ids_arg)
    except Exception as e:
        _print(f"write_*_to_sim failed (non-fatal): {e}")



## hand_release Step-Event entfernt (vereinfachte Lösung über Reset)

