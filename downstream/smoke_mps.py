"""Smoke test: instantiate EEGTransformer + forward pass on MPS with random tensor.
No checkpoint, no dataset. Validates that the encoder runs on Apple Silicon.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from functools import partial
import torch
import torch.nn as nn

from Modules.models.EEGPT_mcae import EEGTransformer
from Modules.Network.utils import Conv1dWithConstraint
from utils import temporal_interpolation

USE_CHANNELS = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']
SLEEP_EDF_INPUT_CHANS = 2
TARGET_SAMPLES = 256 * 30   # 7680

def main():
    has_mps = torch.backends.mps.is_available()
    device = torch.device('mps') if has_mps else torch.device('cpu')
    print(f'device: {device}')

    chans_num = len(USE_CHANNELS)
    encoder = EEGTransformer(
        img_size=[chans_num, TARGET_SAMPLES],
        patch_size=64,
        embed_num=4,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    chan_conv = Conv1dWithConstraint(SLEEP_EDF_INPUT_CHANS, chans_num, 1, max_norm=1)
    chans_id = encoder.prepare_chan_ids(USE_CHANNELS)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'EEGTransformer params: {n_params/1e6:.2f} M')

    encoder = encoder.to(device).eval()
    chan_conv = chan_conv.to(device)
    chans_id = chans_id.to(device)

    B, C, T_RAW = 2, SLEEP_EDF_INPUT_CHANS, 3000
    x = torch.randn(B, C, T_RAW, device=device)
    print(f'input: {tuple(x.shape)}')

    t0 = time.time()
    with torch.no_grad():
        x_interp = temporal_interpolation(x, TARGET_SAMPLES)
        print(f'after interp: {tuple(x_interp.shape)}')
        x_expanded = chan_conv(x_interp)
        print(f'after chan_conv: {tuple(x_expanded.shape)}')
        z = encoder(x_expanded, chans_id)
        print(f'encoder output: {tuple(z.shape)}')
    dt = time.time() - t0
    print(f'forward time: {dt*1000:.0f} ms')

    print('=== SMOKE OK ===')

if __name__ == '__main__':
    main()
