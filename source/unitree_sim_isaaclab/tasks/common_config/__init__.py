"""
公共配置模块
提供可复用的机器人和相机配置
"""

print("[common_config] loaded from:", __file__)

from .robot_configs import G129dofRobotBaseCfg, G1RobotPresets, G1RobotJointTemplates
from .camera_configs import CameraBaseCfg, CameraPresets
from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX

__all__ = [
    "G129dofRobotBaseCfg",
    "G1RobotPresets",
    "G1RobotJointTemplates", 
    "CameraBaseCfg",
    "CameraPresets",
    "G129_CFG_WITH_DEX3_BASE_FIX"
] 