# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from action_provider.action_base import ActionProvider
import math
from typing import Optional
import torch
from typing import Callable


def create_trajectory_generator():
	"""Erzeuge einen Trajektorien-Generator für eine kurze Startbewegung."""
	def trajectory_generator(env, step_count):
		"""Gibt eine direkte Aktion in der Reihenfolge des Action-Terms aus (Mapping-sicher)."""
		robot_data = env.scene["robot"].data

		# DOF-Reihenfolge aus dem Action-Term beziehen
		action_dof_names = None
		try:
			term = env.action_manager.get_term("joint_pos")
			if hasattr(term, "IO_descriptor") and hasattr(term.IO_descriptor, "names"):
				action_dof_names = list(term.IO_descriptor.names)
			elif hasattr(term, "get_IO_descriptors"):
				desc = term.get_IO_descriptors()
				if isinstance(desc, dict) and "names" in desc:
					action_dof_names = list(desc["names"])
		except Exception:
			action_dof_names = None

		if action_dof_names is None:
			action_dof_names = list(robot_data.joint_names)

		num_envs = getattr(env, "num_envs", 1)
		full_action = torch.zeros((num_envs, len(action_dof_names)), device=env.device)

		# Minimal: kein umfangreiches Debug-Log mehr

		# Ziel-DOFs (rechter Arm)
		dof_names_direct = [
			"right_shoulder_pitch_joint",
			"right_elbow_joint",
			"right_wrist_pitch_joint",
		]

		# Feste Zielwinkel für definierte Gelenke (einfach, deterministisch)
		targets = {
			"right_shoulder_pitch_joint": -0.2,
			"right_elbow_joint": 0.4,
			"right_wrist_pitch_joint": 0.3,
		}
		
		# Hand-Release nach 10 Steps: Finger öffnen (früher zum Debuggen)
		hand_release_step = 10
		if step_count >= hand_release_step:
			if step_count == hand_release_step:
				print(f"[ActionDebug] Hand-Release aktiviert bei Step {step_count}")
				# Debug: Alle Roboter-DOFs anzeigen
				robot = env.scene["robot"]
				print(f"[ActionDebug] Roboter hat {len(robot.dof_names)} DOFs")
				print(f"[ActionDebug] Roboter-DOFs: {robot.dof_names}")
				
				# Hand-DOFs finden
				hand_dofs = [dof for dof in robot.dof_names if 'hand' in dof.lower()]
				print(f"[ActionDebug] Hand-DOFs gefunden: {hand_dofs}")
				
				# Ball-Position prüfen
				ball = env.scene["ball"]
				ball_pos = ball.data.root_pos_w[0] if ball.data.root_pos_w.ndim > 1 else ball.data.root_pos_w
				print(f"[ActionDebug] Ball-Position: {ball_pos}")

		# Debug minimal
		if step_count % 60 == 0:
			print(f"[ActionDebug] Step {step_count}")

		name_to_action_idx = {name: i for i, name in enumerate(action_dof_names)}
		actions_set = 0
		
		# Zielwinkel als Residual schreiben: (q_target - q_default)/scale
		try:
			term = env.action_manager.get_term("joint_pos")
			term_scale = float(getattr(term, "scale", getattr(getattr(term, "cfg", object()), "scale", 1.0)))
		except Exception:
			term_scale = 1.0
		q_default = robot_data.default_joint_pos[0] if robot_data.default_joint_pos.ndim == 2 else robot_data.default_joint_pos
		robot_name_to_idx = {n: i for i, n in enumerate(robot_data.joint_names)}
		for dof_name, q_target in targets.items():
			a_idx = name_to_action_idx.get(dof_name)
			r_idx = robot_name_to_idx.get(dof_name)
			if isinstance(a_idx, int) and isinstance(r_idx, int):
				q0 = float(q_default[r_idx])
				residual = (q_target - q0) / max(term_scale, 1e-6)
				residual = max(-1.0, min(1.0, residual))
				full_action[:, a_idx] = residual
				actions_set += 1
		
		# Debug: Prüfen ob Actions gesetzt wurden
		if step_count % 30 == 0:
			print(f"[ActionDebug] Actions gesetzt für {actions_set}/{len(dof_names_direct)} DOFs")
			print(f"[ActionDebug] Action-DOFs verfügbar: {action_dof_names[:10]}...")  # Erste 10
			print(f"[ActionDebug] Gesuchte DOFs: {dof_names_direct}")
			row0 = full_action[0, :10].tolist() if full_action.ndim == 2 else full_action[:10].tolist()
			print(f"[ActionDebug] Vollständige Action[env0]: {row0}...")  # Erste 10 Werte
			
			# Debug reduziert

		return full_action
	
	return trajectory_generator

class TrajectoryActionProvider(ActionProvider):
    """基于轨迹生成的Action提供者"""
    
    def __init__(self, trajectory_generator: Callable):
        super().__init__("TrajectoryActionProvider")
        self.trajectory_generator = trajectory_generator
        self.step_count = 0
    
    def get_action(self, env) -> Optional[torch.Tensor]:
        """从轨迹生成器获取action"""
        try:
            action = self.trajectory_generator(env, self.step_count)
            self.step_count += 1
            return action
        except Exception as e:
            print(f"[{self.name}] 获取轨迹action失败: {e}")
            return None