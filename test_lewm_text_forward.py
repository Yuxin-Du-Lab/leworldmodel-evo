import sys
from types import SimpleNamespace

import torch

sys.path.append('D:/work/worldmodel/world_model/le-wm-main')

from jepa import JEPA
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


def main():
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
        model_name='openai/clip-vit-base-patch32',
        output_dim=embed_dim,
        freeze=True,
        max_length=32,
    )
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=256, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=256, norm_fn=torch.nn.BatchNorm1d)

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

    batch = {
        'pixels': torch.randn(2, 2, 3, 224, 224),
        'text': ['pick up the bowl', 'open the drawer'],
    }

    output = lejepa_forward(wrapper, batch, 'train', cfg)

    assert output['emb'].shape == (2, 2, embed_dim)
    assert output['text_emb'].shape == (2, embed_dim)
    assert output['pred_emb'].shape == (2, 1, embed_dim)
    assert output['loss'].ndim == 0
    print('text-conditioned forward smoke test passed')


if __name__ == '__main__':
    main()
