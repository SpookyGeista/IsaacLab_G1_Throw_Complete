# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Observation functions for G1 Throwing Test Task"""

from unitree_sim_isaaclab.tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from unitree_sim_isaaclab.tasks.common_observations.dex3_state import get_robot_dex3_joint_states
from unitree_sim_isaaclab.tasks.common_observations.camera_state import get_camera_image
import torch

def get_hand_position(env):
    """Temporary function to get right hand position and print it"""
    try:
        # Get the robot
        robot = env.scene["robot"]
        
        # Get all body names
        body_names = robot.data.body_names
        print(f"Available body names: {body_names}")
        
        # Look for right hand links
        right_hand_links = [name for name in body_names if "right_hand" in name.lower()]
        print(f"Right hand links found: {right_hand_links}")
        
        if right_hand_links:
            # Get the first right hand link position
            link_name = right_hand_links[0]
            link_idx = body_names.index(link_name)
            hand_pos = robot.data.body_pos_w[0, link_idx]  # [0] for first environment
            print(f"Position of {link_name}: {hand_pos}")
            return hand_pos
        else:
            print("No right hand links found!")
            return torch.zeros(3)
    except Exception as e:
        print(f"Error getting hand position: {e}")
        return torch.zeros(3)

# ensure functions can be accessed by external modules
__all__ = [
    "get_robot_boy_joint_states",
    "get_robot_dex3_joint_states", 
    "get_camera_image",
    "get_hand_position"
]
