"""Time one EEGPT training step under the SADT input geometry.

Run from `downstream/`. Uses random tensors, so it needs neither the pretrained
checkpoint nor the dataset -- the point is the cost of a step on this GPU, which
sets the budget for the Leave-One-Subject-Out campaign.
"""
import sys
import time
from functools import partial

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from Modules.models.EEGPT_mcae import EEGTransformer  # noqa: E402
from Modules.Network.utils import Conv1dWithConstraint  # noqa: E402

# SADT: 4 s at 256 Hz over 30 scalp channels. The montage uses the old 10-20
# names T3/T4/T5/T6, which EEGPT's channel vocabulary does not carry, so they are
# renamed to their 10-10 equivalents T7/T8/P7/P8 before lookup.
USE_CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ',
                'FC4', 'FT8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'TP7', 'CP3', 'CPZ',
                'CP4', 'TP8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']

CH_IN, T, PATCH, BATCH = 30, 1024, 64, 32

d = torch.device("cuda")
chans_num = len(USE_CHANNELS)

encoder = EEGTransformer(
    img_size=[chans_num, T], patch_size=PATCH, embed_num=4, embed_dim=512,
    depth=8, num_heads=8, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0,
    drop_path_rate=0.0, init_std=0.02, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))
chan_conv = Conv1dWithConstraint(CH_IN, chans_num, 1, max_norm=1)
chans_id = encoder.prepare_chan_ids(USE_CHANNELS).to(d)

n_params = sum(p.numel() for p in encoder.parameters())
print("encoder params: %.2f M" % (n_params / 1e6))
print("channels: %d in -> %d encoder" % (CH_IN, chans_num))

encoder = encoder.to(d)
chan_conv = chan_conv.to(d)
x = torch.randn(BATCH, CH_IN, T, device=d)


def bench(amp, steps=12):
    """Two warm-up steps, then average over the rest."""
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(steps):
        if i == 2:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            z = encoder(chan_conv(x), chans_id)
            loss = z.float().pow(2).mean()
        loss.backward()
        encoder.zero_grad(set_to_none=True)
        chan_conv.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / (steps - 2)
    return tuple(z.shape), dt, torch.cuda.max_memory_allocated() / 1e9


shape, dt32, mem32 = bench(False)
print("output shape:", shape)
print("fp32 fwd+bwd batch=%d: %.1f ms/step, peak %.1f GB" % (BATCH, dt32 * 1000, mem32))

shape, dt16, mem16 = bench(True)
print("fp16 fwd+bwd batch=%d: %.1f ms/step, peak %.1f GB" % (BATCH, dt16 * 1000, mem16))

cap = torch.cuda.get_device_capability(0)
sm = "sm_%d%d" % cap
print("RESULT_BEGIN")
print("gpu=%s" % torch.cuda.get_device_name(0))
print("capability=%s" % sm)
print("native_kernels=%s" % (sm in torch.cuda.get_arch_list()))
print("encoder_params_m=%.2f" % (n_params / 1e6))
print("output_shape=%s" % (shape,))
print("ms_per_step_fp32=%.1f" % (dt32 * 1000))
print("ms_per_step_fp16=%.1f" % (dt16 * 1000))
print("peak_gb_fp16=%.1f" % mem16)
print("RESULT_END")
