# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Simple Throwing Scene Configuration - Unitree Style"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import CameraBaseCfg, G1RobotPresets

@configclass
class SimpleThrowSceneCfg(InteractiveSceneCfg):
    """Simple Throwing Scene Configuration - Unitree Style"""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",    # ground in the scene
        spawn=GroundPlaneCfg(),    # ground configuration
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )

    # G1 Robot with Dex3 hand - using throw-optimized preset
    robot = G1RobotPresets.g1_29dof_dex3_throw_base_fix(
        init_pos=(-0.15, 0.0, 0.76),  # robot position
        init_rot=(0.7071, 0, 0, 0.7071)  # robot rotation
    )

    # Tennis Ball for throwing - positioned in robot's right hand
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",    # ball in the scene
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.00050026, 0.23872, 0.84455],  # exact position of right_hand_camera_base_link
            rot=[1, 0, 0, 0]  # initial rotation
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.03,  # tennis ball radius (3 cm)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.057),  # tennis ball mass (57g)
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.8, 0.0),  # yellow color
                metallic=0.0,
                roughness=0.8
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=0.6,  # tennis ball friction
                dynamic_friction=0.5,
                restitution=0.7,  # tennis ball bounce
            ),
        ),
    )

    # World Camera
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-0.1, 3.6, 1.6),
        rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
        focal_length=16.5,
    )
