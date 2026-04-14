import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'le-wm-main'))

from train_lerobot_text import build_dataloaders, build_lerobot_dataset, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='le-wm-main/config/train/lewm_lerobot_text.yaml')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = build_lerobot_dataset(cfg)
    assert len(dataset) > 0, 'dataset is empty'

    data_module = build_dataloaders(dataset, cfg)
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    assert 'pixels' in batch
    assert 'text' in batch
    assert batch['pixels'].ndim == 5
    assert len(batch['text']) == batch['pixels'].shape[0]

    module = build_model(cfg)
    assert module is not None

    print('train_lerobot_text setup test passed')


if __name__ == '__main__':
    main()
