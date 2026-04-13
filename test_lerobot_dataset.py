import argparse
import json
from pathlib import Path

import torch
import yaml

from lerobot_dataset_pretrain_mp import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_libero.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max-samples-per-file", type=int, default=None)
    parser.add_argument("--use-augmentation", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = LeRobotDataset(
        config=config,
        image_size=args.image_size,
        action_horizon=args.action_horizon,
        cache_dir=args.cache_dir,
        max_samples_per_file=args.max_samples_per_file,
        use_augmentation=args.use_augmentation,
    )

    print(f"dataset length: {len(dataset)}")
    sample = dataset[args.index]

    print("keys:", sorted(sample.keys()))
    print("prompt:", sample["prompt"])
    print("text:", sample["text"])
    print("text == prompt:", sample["text"] == sample["prompt"])

    tensor_keys = [
        "images",
        "image_mask",
        "state",
        "state_mask",
        "action",
        "action_mask",
        "embodiment_id",
        "pixels",
    ]
    for key in tensor_keys:
        value = sample[key]
        print(f"{key}: shape={tuple(value.shape) if hasattr(value, 'shape') else 'scalar'}, dtype={getattr(value, 'dtype', type(value))}")

    assert "pixels" in sample
    assert "text" in sample
    assert sample["text"] == sample["prompt"]
    assert sample["pixels"].ndim == 4
    assert sample["pixels"].shape[0] == 2
    assert sample["pixels"].shape[1] == 3
    assert sample["pixels"].shape[2] == args.image_size
    assert sample["pixels"].shape[3] == args.image_size

    required_old_keys = {
        "images",
        "image_mask",
        "prompt",
        "state",
        "state_mask",
        "action",
        "action_mask",
        "embodiment_id",
    }
    missing = required_old_keys - set(sample.keys())
    assert not missing, f"missing old keys: {missing}"

    print("dataset sample check passed")


if __name__ == "__main__":
    main()
