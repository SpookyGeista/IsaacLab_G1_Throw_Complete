# Recovered throwing policy

## Files and provenance

`checkpoints/throw_2025-10-01/` contains:

| File | Purpose |
| --- | --- |
| `model_21999.pt` | Last numbered checkpoint found in training run `2025-10-01_13-10-59` |
| `actor_21999.pt` | TorchScript actor exported directly from that checkpoint |
| `env.yaml` | Saved training environment configuration; local asset path made portable |
| `agent.yaml` | Saved PPO configuration |
| `manifest.json` | Checksums, architecture, source run, and verification details |

The historical experiment directory was named `franka_lift`. Its saved environment references the G1 throwing task, the ball-release events, and the three right-arm joints. This identifies it as a throwing run despite the inherited directory name. Iteration 21999 is the last recovered checkpoint, not a claim that it was the best policy or the exact checkpoint used in the reported hardware trial.

An older `exported/policy.pt` also exists locally. Its actor weights differ from this checkpoint, so the published export is rebuilt from `model_21999.pt`.

## Model interface

The actor is an ELU MLP: `101 → 256 → 128 → 64 → 3`, with no empirical observation normalization in the saved agent configuration. Inputs and outputs use float32.

The saved configuration concatenates `robot_joint_state` followed by `robot_dex3_state`. The historical observation helpers still present in this repository implement:

| Slice | Signal |
| --- | --- |
| `0:29` | Body joint positions |
| `29:58` | Body joint velocities |
| `58:87` | Body applied torques |
| `87:101` | Dex3 finger positions |

The body index order in those helpers is `[0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]`. The finger order is `[31,37,41,30,36,29,35,34,40,42,33,39,32,38]`. These indices refer to the historical simulator articulation; they are not hardware motor IDs. Confirm the articulation's joint names and order before using the model. Matching the vector length alone is insufficient.

The action term selects `right_shoulder_pitch_joint`, `right_elbow_joint`, and `right_wrist_yaw_joint`. The saved scale is `0.35`; the elbow offset is `2.0944` rad. Saved clipping ranges are shoulder `[-0.6, 0.6]`, elbow `[-1, 1]`, and wrist yaw `[-0.5, 0.5]`. `preserve_order` is false: confirm resolved articulation order and Isaac Lab's action-processing semantics before interpreting individual output columns. The inference tool prints raw actor outputs without applying joint targets.

## Setup and tools

Follow the CPU quick start in the root README. The supplied tool loads the checkpoint with `weights_only=True`, checks the actor architecture, and validates finite outputs. `--export` creates a TorchScript actor and checks exact equality against the checkpoint actor on the supplied observations. It does not publish DDS messages or send motor commands.

```python
import torch

actor = torch.jit.load("checkpoints/throw_2025-10-01/actor_21999.pt", map_location="cpu").eval()
with torch.inference_mode():
    raw_actions = actor(torch.zeros(1, 101))
print(raw_actions.shape)  # torch.Size([1, 3]); synthetic input only
```

## Restoring simulator playback

Use the saved `env.yaml` and `agent.yaml` as the record of the later training setup, rather than the earlier all-joint configuration in this repository. Important saved settings include physics step `0.005 s`, decimation `6`, a 10-second episode, and height/velocity-triggered ball release. The agent uses PPO, seed 42, and 24 steps per environment.

The original callable paths use `isaaclab_tasks.tasks.*`. That modified training checkout has not been recovered in full. The saved YAML contains Python-specific serialization tags and is a configuration record, not a standalone environment installer. Reconstructing it requires the matching reset/release functions, reward functions, robot asset, observation ordering, and task registration before using Isaac Lab's training or playback scripts. Full simulator reproduction has not been verified.

The local asset path in the saved YAML has been replaced by `assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd`. Resolve it against the repository root when restoring the environment. The current local port changes observations and the wrist offset; it has deliberately not been presented as the original training environment.

## Training curves

`results/training/` retains the original CSV exports for mean reward, ball-height reward, and target-distance reward. Columns are `step,tag,value`. They describe the recovered run; reward values are not independent throw-success measurements.

```python
import csv
from pathlib import Path

for path in sorted(Path("results/training").glob("*.csv")):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    print(path.name, len(rows), rows[0]["step"], rows[-1]["step"])
```
