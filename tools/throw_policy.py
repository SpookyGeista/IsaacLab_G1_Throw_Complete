"""Export and exercise the archived G1 throw actor without a simulator or DDS."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "checkpoints/throw_2025-10-01/model_21999.pt"


def load_actor(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved["model_state_dict"]
    actor = nn.Sequential(
        nn.Linear(101, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 3),
    )
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items()
                           if k.startswith("actor.")}, strict=True)
    return actor.eval(), int(saved["iter"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT)
    parser.add_argument("--export", type=Path, help="Write a TorchScript actor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Use synthetic observations")
    group.add_argument("--observations", type=Path, help="JSON array with shape [N, 101]")
    args = parser.parse_args()
    actor, iteration = load_actor(args.checkpoint)
    if args.smoke_test:
        generator = torch.Generator().manual_seed(42)
        observations = torch.cat((torch.zeros(1, 101), torch.randn(15, 101, generator=generator)))
    else:
        observations = torch.tensor(json.loads(args.observations.read_text()), dtype=torch.float32)
    if observations.ndim != 2 or observations.shape[1] != 101 or observations.shape[0] == 0:
        parser.error("observations must have shape [N, 101], with N >= 1")
    if not torch.isfinite(observations).all():
        parser.error("observations contain non-finite values")
    with torch.inference_mode():
        actions = actor(observations)
        if not torch.isfinite(actions).all():
            raise ValueError("actor produced non-finite actions")
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            torch.jit.script(actor).save(str(args.export))
            restored = torch.jit.load(str(args.export), map_location="cpu").eval()
            torch.testing.assert_close(restored(observations), actions, rtol=0, atol=0)
    print(json.dumps({"iteration": iteration, "input_shape": list(observations.shape),
                      "output_shape": list(actions.shape), "synthetic": args.smoke_test,
                      "raw_actions": actions.tolist()}, indent=2))


if __name__ == "__main__":
    main()
