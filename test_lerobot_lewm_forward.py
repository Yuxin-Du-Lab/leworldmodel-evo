import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "le-wm-main"))

from jepa import JEPA
from lerobot_dataset_pretrain_mp import LeRobotDataset
from module import ARPredictor, CLIPTextConditioner, MLP, SIGReg
from train import lejepa_forward


class DummyVisionOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class DummyVisionEncoder(torch.nn.Module):
    def __init__(self, hidden_size=192):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, interpolate_pos_encoding=True):
        b = x.shape[0]
        cls = torch.randn(b, 1, self.hidden_size, device=x.device)
        return DummyVisionOutput(last_hidden_state=cls)


def build_batch(dataset, batch_size):
    samples = [dataset[i] for i in range(batch_size)]
    batch = {
        "pixels": torch.stack([sample["pixels"] for sample in samples], dim=0),
        "text": [sample["text"] for sample in samples],
    }
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_libero.yaml")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples-per-file", type=int, default=None)
    args = parser.parse_args()

    with open(Path(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = LeRobotDataset(
        config=config,
        image_size=args.image_size,
        action_horizon=args.action_horizon,
        cache_dir=args.cache_dir,
        max_samples_per_file=args.max_samples_per_file,
    )

    assert len(dataset) >= args.batch_size, f"dataset too small: len={len(dataset)}"

    batch = build_batch(dataset, args.batch_size)
    print("batch['pixels']:", batch["pixels"].shape, batch["pixels"].dtype)
    print("len(batch['text']):", len(batch["text"]))
    print("batch['text'][0]:", batch["text"][0])

    hidden_dim = 192
    embed_dim = 192

    encoder = DummyVisionEncoder(hidden_size=hidden_dim)
    predictor = ARPredictor(
        num_frames=1,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        depth=2,
        heads=4,
        mlp_dim=256,
        dim_head=32,
        dropout=0.0,
        emb_dropout=0.0,
    )
    text_encoder = CLIPTextConditioner(
        model_name="openai/clip-vit-base-patch32",
        output_dim=embed_dim,
        freeze=True,
        max_length=32,
    )
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=256,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=256,
        norm_fn=torch.nn.BatchNorm1d,
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        text_encoder=text_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )

    wrapper = SimpleNamespace(
        model=model,
        sigreg=SIGReg(knots=17, num_proj=32),
        log_dict=lambda *args, **kwargs: None,
    )

    cfg = SimpleNamespace(
        wm=SimpleNamespace(history_size=1, num_preds=1),
        loss=SimpleNamespace(sigreg=SimpleNamespace(weight=0.09)),
    )

    output = lejepa_forward(wrapper, batch, "train", cfg)

    print("emb:", output["emb"].shape)
    print("text_emb:", output["text_emb"].shape)
    print("pred_emb:", output["pred_emb"].shape)
    print("loss:", output["loss"].shape if hasattr(output["loss"], "shape") else type(output["loss"]))

    assert batch["pixels"].shape == (args.batch_size, 2, 3, args.image_size, args.image_size)
    assert len(batch["text"]) == args.batch_size
    assert output["emb"].shape == (args.batch_size, 2, embed_dim)
    assert output["text_emb"].shape == (args.batch_size, embed_dim)
    assert output["pred_emb"].shape == (args.batch_size, 1, embed_dim)
    assert output["loss"].ndim == 0

    print("real dataset + text-conditioned forward test passed")


if __name__ == "__main__":
    main()
