import gymnasium as gym

# Lazy-Referenz auf die Cfg-Klasse als String, damit keine Imports nötig sind
_ENV_CFG_ENTRY_POINT = (
    "unitree_sim_isaaclab.tasks.g1_tasks.g1_throw_test.g1_throw_test_env_cfg:"
    "G1ThrowTestEnvCfg"
)

# Registriert die Env-ID beim Import dieses Moduls (ohne schwere Imports)
gym.register(
    id="Isaac-G1-Throw-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": _ENV_CFG_ENTRY_POINT},
    disable_env_checker=True,
)
