from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import yaml
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from jepa import JEPA
from lerobot_dataset_pretrain_mp import LeRobotDataset
from module import ARPredictor, CLIPTextConditioner, MLP, SIGReg
from train import lejepa_forward
from utils import ModelObjectCallBack


def build_lerobot_dataset(cfg):
    with open(cfg.lerobot.config_path, 'r', encoding='utf-8') as f:
        dataset_cfg = yaml.safe_load(f)

    dataset = LeRobotDataset(
        config=dataset_cfg,
        image_size=cfg.lerobot.image_size,
        action_horizon=cfg.lerobot.action_horizon,
        cache_dir=cfg.lerobot.cache_dir,
        max_samples_per_file=cfg.lerobot.max_samples_per_file,
        use_augmentation=cfg.lerobot.use_augmentation,
    )
    return dataset


def build_dataloaders(dataset, cfg):
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    return spt.data.DataModule(train=train, val=val)


def build_model(cfg):
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get('embed_dim', hidden_dim)

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    text_encoder = CLIPTextConditioner(
        model_name=cfg.text_encoder.model_name,
        output_dim=embed_dim,
        freeze=cfg.text_encoder.freeze,
        max_length=cfg.text_encoder.max_length,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        text_encoder=text_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {'type': 'LinearWarmupCosineAnnealingLR'},
            'interval': 'epoch',
        },
    }

    return spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )


@hydra.main(version_base=None, config_path='./config/train', config_name='lewm_lerobot_text')
def run(cfg):
    dataset = build_lerobot_dataset(cfg)
    data_module = build_dataloaders(dataset, cfg)
    world_model = build_model(cfg)

    run_id = cfg.get('subdir') or ''
    base_dir = (
        Path(cfg.output_root_dir)
        if cfg.output_root_dir
        else Path(swm.data.utils.get_cache_dir())
    )
    run_dir = base_dir / run_id if run_id else base_dir

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(save_dir=str(run_dir), **cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / 'checkpoints',
        filename='lewm-lerobot-{epoch:02d}-{step}',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, checkpoint_callback],
        logger=logger,
        enable_checkpointing=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f'{cfg.output_model_name}_weights.ckpt',
    )
    manager()


if __name__ == '__main__':
    run()
