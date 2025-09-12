# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Unitree Sim IsaacLab - G1 robot simulation tasks for Isaac Lab."""

__version__ = "0.1.0"

# Bei Paketimport direkt die Task-Registrierung ausführen
try:
    import unitree_sim_isaaclab.tasks.g1_tasks.g1_throw_test.register  # noqa: F401
except Exception:
    # Registrierung erfolgt dann, sobald die Abhängigkeiten verfügbar sind
    pass
