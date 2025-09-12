#!/usr/bin/env python3
"""
Debug-Script für das Reset-Problem
Überwacht Episode-Länge und Termination-Status
"""

import time
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_episode_info(env, step_i):
    """Debug-Funktion für Episode-Informationen"""
    if step_i % 60 == 0:  # Every 60 steps
        print("=== DEBUG INFO ===")
        print(f"Step: {step_i}")
        print(f"Episode length buf: {int(env.episode_length_buf[0])}")
        print(f"Max episode length: {int(env.max_episode_length)}")
        print(f"Step dt: {float(env.step_dt)}")
        print(f"Sim time: {float(env.episode_length_buf[0]) * float(env.step_dt):.2f}s")
        print(f"Terminated: {bool(env.termination_manager.terminated.any())}")
        print(f"Time outs: {bool(env.termination_manager.time_outs.any())}")
        print("==================")

if __name__ == "__main__":
    print("Debug-Script für Reset-Problem")
    print("Dieses Script kann in die Simulation integriert werden")
