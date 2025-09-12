# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import EventTermCfg
from unitree_sim_isaaclab.tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from . import mdp

# import our simple scene configuration
from unitree_sim_isaaclab.tasks.common_scene.simple_throw_scene_cfg import SimpleThrowSceneCfg

# Import the working example as base
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Scene definition
##

@configclass
class G1ThrowTestSceneCfg(SimpleThrowSceneCfg):
    """Einfache Wurf-Szene: Boden, Licht, G1(Dex3)."""
    env_spacing: float = 2.5
    

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Direkte Gelenkwinkelregelung für den Roboter (für Policy-Training)."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=10.0,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Beobachtungen für die Policy."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy‑Gruppe (states + Kamera)."""
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_dex3_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        # camera_image = ObsTerm(func=mdp.get_camera_image)  # Kameras deaktiviert für Stabilität

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()

# Import terminations from separate file
from .mdp.terminations import TerminationsCfg
from .mdp import rewards as rw

@configclass
class EventCfg:
    # Reset-Event
    reset_ball_and_robot = EventTermCfg(func=mdp.reset_ball_and_robot, mode="reset")

##
# Environment configuration
##

@configclass
class G1ThrowTestEnvCfg(ManagerBasedRLEnvCfg):
    """G1 Throwing Test Environment Configuration"""
    
    # scene configuration
    scene: G1ThrowTestSceneCfg = G1ThrowTestSceneCfg()
    
    # MDP configuration
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    @configclass
    class RewardsCfg:
        # Belohne Ballhöhe leicht
        ball_height = RewardTermCfg(func=rw.reward_ball_height, weight=1.0)
        # Belohne horizontale Distanz um Zielradius (Gauss-Shaping)
        ball_xy_gauss = RewardTermCfg(
            func=rw.reward_ball_xy_distance_gauss,
            weight=1.0,
            params={"target_xy_dist": 1.0, "sigma": 0.25, "scale": 1.0},
        )

    rewards: RewardsCfg = RewardsCfg()
    
    # environment settings
    env_spacing: float = 2.5
    episode_length_s: float = 10.0  # Required by IsaacLab
    num_envs: int = 2  # zwei Environments zum Testen von env_origins
    decimation: int = 4  # control frequency decimation
    
    # physics settings
    physics_dt: float = 0.005
    rendering_dt: float = 0.02
    control_frequency_inv: int = 1
    
    # debug settings
    debug_vis: bool = True
    enable_camera_sensors: bool = False  # Kameras deaktiviert für Stabilität
    
    # use default finite horizon setting
    # is_finite_horizon: bool = False  # default
    
    def __post_init__(self):
        """Post initialization - CRITICAL for reset functionality!"""
        # simulation settings
        self.sim.dt = self.physics_dt
        self.sim.render_interval = self.decimation
        
        # Episodenlänge in Steps: episode_length_s / (physics_dt * decimation) == ~10s/0.02 -> 500
        self.max_episode_length = int(self.episode_length_s / (self.physics_dt * self.decimation))
        
        # CRITICAL: Set is_finite_horizon to True for automatic reset
        self.is_finite_horizon = True

        # Sichtbarer Refresh beim Reset (optional, hilft bei Feedback)
        self.rerender_on_reset = True
