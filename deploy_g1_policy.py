#!/usr/bin/env python3
"""
Deploy script for G1 Policy trained with Isaac Lab
This script loads the trained policy and deploys it to the real G1 robot
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
from pathlib import Path

# Add Isaac Lab to path
isaaclab_path = Path(__file__).parent.parent
sys.path.append(str(isaaclab_path))

# Unitree SDK2 imports
try:
    from unitree_sdk2py import unitree_sdk2py
    from unitree_sdk2py import high_level
    from unitree_sdk2py import low_level
except ImportError:
    print("Unitree SDK2 Python not found. Please install it first:")
    print("pip install unitree_sdk2py")
    sys.exit(1)

# Isaac Lab imports
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


class G1PolicyDeployer:
    """Deployer for G1 policies trained with Isaac Lab"""
    
    def __init__(self, policy_path: str, network_interface: str = "enp2s0"):
        """
        Initialize the policy deployer
        
        Args:
            policy_path: Path to the trained policy file (.pt)
            network_interface: Network interface to connect to G1
        """
        self.policy_path = Path(policy_path)
        self.network_interface = network_interface
        
        # Load policy
        self.policy = self._load_policy()
        
        # Initialize G1 connection
        self.g1_client = self._init_g1_connection()
        
        # Initialize observation processor
        self.obs_processor = self._init_observation_processor()
        
        print(f"Policy loaded from: {policy_path}")
        print(f"Connected to G1 via: {network_interface}")
    
    def _load_policy(self):
        """Load the trained policy"""
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")
        
        # Load the policy state dict
        checkpoint = torch.load(self.policy_path, map_location='cpu')
        
        # Extract policy state dict
        if 'actor' in checkpoint:
            policy_state_dict = checkpoint['actor']
        elif 'model' in checkpoint:
            policy_state_dict = checkpoint['model']
        else:
            policy_state_dict = checkpoint
        
        # Create policy network with same architecture as training
        policy_cfg = G1FlatPPORunnerCfg().policy
        policy = self._create_policy_network(policy_cfg)
        
        # Load weights
        policy.load_state_dict(policy_state_dict)
        policy.eval()
        
        return policy
    
    def _create_policy_network(self, policy_cfg: RslRlPpoActorCriticCfg):
        """Create policy network with same architecture as training"""
        from rsl_rl.modules import ActorCritic
        
        # Create actor-critic network
        actor_critic = ActorCritic(
            num_obs=64,  # G1 observation dimension
            num_actions=12,  # G1 action dimension (12 DOF)
            actor_hidden_dims=policy_cfg.actor_hidden_dims,
            critic_hidden_dims=policy_cfg.critic_hidden_dims,
            activation=policy_cfg.activation,
            init_noise_std=policy_cfg.init_noise_std,
        )
        
        return actor_critic.actor
    
    def _init_g1_connection(self):
        """Initialize connection to G1 robot"""
        try:
            # Initialize high-level client
            client = high_level.HighLevelClient(self.network_interface)
            print("Connected to G1 high-level interface")
            return client
        except Exception as e:
            print(f"Failed to connect to G1: {e}")
            print("Make sure the robot is powered on and connected to the network")
            sys.exit(1)
    
    def _init_observation_processor(self):
        """Initialize observation processor for G1 state"""
        # This would need to be customized based on your specific observation space
        # For now, we'll use a simple processor
        return None
    
    def _get_g1_state(self):
        """Get current state from G1 robot"""
        try:
            # Get high-level state
            state = self.g1_client.get_high_level_state()
            
            # Extract relevant observations
            obs = self._process_g1_state(state)
            return obs
            
        except Exception as e:
            print(f"Failed to get G1 state: {e}")
            return None
    
    def _process_g1_state(self, state):
        """Process G1 state into observation vector"""
        # This needs to match your training observation space
        # For G1 velocity control, typical observations include:
        # - Base linear velocity (3D)
        # - Base angular velocity (3D)
        # - Joint positions (12 DOF)
        # - Joint velocities (12 DOF)
        # - Command velocity (3D)
        
        obs = []
        
        # Base velocity (6D)
        obs.extend([state.base_velocity.x, state.base_velocity.y, state.base_velocity.z])
        obs.extend([state.base_angular_velocity.x, state.base_angular_velocity.y, state.base_angular_velocity.z])
        
        # Joint positions (12D)
        for joint in state.joint_state:
            obs.append(joint.q)
        
        # Joint velocities (12D)
        for joint in state.joint_state:
            obs.append(joint.dq)
        
        # Command velocity (3D) - you might need to set this based on desired velocity
        # For now, using current velocity as command
        obs.extend([state.base_velocity.x, state.base_velocity.y, state.base_angular_velocity.z])
        
        # Add any additional observations your policy expects
        # This might include IMU data, foot contact states, etc.
        
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    def _send_action_to_g1(self, action):
        """Send action to G1 robot"""
        try:
            # Convert action to joint commands
            joint_commands = self._action_to_joint_commands(action)
            
            # Send high-level command
            self.g1_client.set_high_level_command(joint_commands)
            
        except Exception as e:
            print(f"Failed to send action to G1: {e}")
    
    def _action_to_joint_commands(self, action):
        """Convert policy action to G1 joint commands"""
        # This depends on your action space definition
        # For G1, actions are typically joint position targets or joint velocity targets
        
        # Convert action tensor to numpy
        action_np = action.squeeze().cpu().numpy()
        
        # Create joint command structure
        # You'll need to adjust this based on your specific action space
        joint_commands = high_level.HighLevelCommand()
        
        # Set joint positions (assuming action is joint position targets)
        for i, pos in enumerate(action_np):
            if i < len(joint_commands.joint_command):
                joint_commands.joint_command[i].q = pos
        
        return joint_commands
    
    def deploy(self, duration: float = 60.0, control_freq: float = 50.0):
        """
        Deploy the policy to G1 robot
        
        Args:
            duration: Duration to run the policy (seconds)
            control_freq: Control frequency (Hz)
        """
        print(f"Starting policy deployment for {duration} seconds at {control_freq} Hz")
        
        dt = 1.0 / control_freq
        num_steps = int(duration * control_freq)
        
        try:
            for step in range(num_steps):
                start_time = time.time()
                
                # Get current state
                obs = self._get_g1_state()
                if obs is None:
                    print("Failed to get state, skipping step")
                    continue
                
                # Get action from policy
                with torch.no_grad():
                    action = self.policy(obs)
                
                # Send action to robot
                self._send_action_to_g1(action)
                
                # Control timing
                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                
                if step % 100 == 0:
                    print(f"Step {step}/{num_steps}")
                
        except KeyboardInterrupt:
            print("\nPolicy deployment interrupted by user")
        except Exception as e:
            print(f"Error during deployment: {e}")
        finally:
            self._stop_robot()
    
    def _stop_robot(self):
        """Stop the robot safely"""
        try:
            # Send stop command
            stop_command = high_level.HighLevelCommand()
            self.g1_client.set_high_level_command(stop_command)
            print("Robot stopped safely")
        except Exception as e:
            print(f"Failed to stop robot: {e}")


def main():
    parser = argparse.ArgumentParser(description="Deploy G1 Policy")
    parser.add_argument("--policy_path", type=str, required=True,
                       help="Path to trained policy file (.pt)")
    parser.add_argument("--network_interface", type=str, default="enp2s0",
                       help="Network interface to connect to G1")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Duration to run policy (seconds)")
    parser.add_argument("--control_freq", type=float, default=50.0,
                       help="Control frequency (Hz)")
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = G1PolicyDeployer(
        policy_path=args.policy_path,
        network_interface=args.network_interface
    )
    
    # Deploy policy
    deployer.deploy(duration=args.duration, control_freq=args.control_freq)


if __name__ == "__main__":
    main() 