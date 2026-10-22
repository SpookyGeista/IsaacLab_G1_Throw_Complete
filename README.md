# Unitree G1 Throwing Policy

Reinforcement learning experiments on a fixed-base Unitree G1: a three-joint throwing policy, reward design, and a follow-up analysis of joint loads during sim-to-real transfer. Work by Lorenz Valentin Pflanz.

## Included work

- Throwing task and ball-release logic in `source/unitree_sim_isaaclab/tasks/g1_tasks/g1_throw_test/`.
- Recovered training checkpoint from the October 2025 experiment, plus a matching TorchScript actor.
- Training configuration snapshots and three original training curves as CSV files.
- [Results](docs/RESULTS.md) from the bachelor thesis and the subsequent joint-load study.
- [Policy setup and usage](docs/THROW_POLICY.md), including the observation format and compatibility limits.

The original simulator documentation is retained in [README_UNITREE.md](README_UNITREE.md). The archived policy comes from a later training configuration than the task code currently in this repository.

## Quick start: inspect the recovered policy

This path runs on a CPU and does not require Isaac Sim or connect to a robot. Use Python 3.11 and PyTorch 2.6 or newer; the export was checked with the version recorded in the checkpoint manifest.

```bash
git clone https://github.com/SpookyGeista/IsaacLab_G1_Throw_Complete.git
cd IsaacLab_G1_Throw_Complete
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.6,<3"
python tools/throw_policy.py --smoke-test
```

Expected: checkpoint iteration `21999`, input shape `[16, 101]`, output shape `[16, 3]`, and finite raw actions. Synthetic observations check model loading and inference; they do not test throwing performance.

To rebuild and check the TorchScript export:

```bash
python tools/throw_policy.py --smoke-test --export /tmp/g1_throw_actor.pt
```

For recorded observations, supply a JSON array with one 101-value vector per row:

```bash
python tools/throw_policy.py --observations /path/to/observations.json
```

See [THROW_POLICY.md](docs/THROW_POLICY.md) before connecting the actor to a simulation. Raw network outputs require the original action mapping.

## Simulation and hardware status

The thesis used Isaac Sim 5.0 and a modified Isaac Lab training environment. The recovered snapshot specifies three right-arm actions and concatenated observations. The earlier task in this repository uses a different action configuration and is not a compatible playback environment for this checkpoint without restoration of the later task configuration and its supporting functions.

The hardware trial was stopped after excessive wrist loading; the follow-up report investigates an actuator failure. The policy is published as a research artifact. No validated hardware deployment procedure is supplied for this recovered checkpoint.

## Related project

[G1 Hand Positioning RL](https://github.com/SpookyGeista/g1-hand-positioning-rl) contains the subsequent hand-positioning work compared in the load study.

## Foundations

Built on [Unitree's simulator integration](https://github.com/unitreerobotics/unitree_sim_isaaclab), [Isaac Lab](https://github.com/isaac-sim/IsaacLab), Isaac Sim, and RSL-RL. Existing copyright notices and licenses remain applicable; see [LICENSE](LICENSE) and the original simulator README.
