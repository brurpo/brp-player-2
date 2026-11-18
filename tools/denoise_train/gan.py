"""The adversarial test (train.py --gan-weight): a critic trained beside the denoiser to
tell its output from clean frames, so the denoiser may put plausible fine texture where
the exact one cannot be known (the pixel losses alone answer that with blur).

The critic sees single frames only (crops), nothing of time: that invented texture stays
the same from frame to frame is the steadiness loss's job (train.stability: the output
may only change along the motion as the clean frames do), so --gan-weight goes with a
raised --stab-weight. Trained from scratch here, nothing pretrained.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """A patch critic: 8 convolutions down to 1/8, about 45 px of the picture per score
    (texture, not content). Spectral normalisation keeps it from running away."""

    def __init__(s, w=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        chans = [(3, w, 3, 1), (w, w, 4, 2), (w, 2 * w, 3, 1), (2 * w, 2 * w, 4, 2), (2 * w, 4 * w, 3, 1), (4 * w, 4 * w, 4, 2), (4 * w, 4 * w, 3, 1)]
        layers = []
        for i, o, k, st in chans:
            layers += [sn(nn.Conv2d(i, o, k, st, 1)), nn.LeakyReLU(0.2, True)]
        s.net = nn.Sequential(*layers, nn.Conv2d(4 * w, 1, 3, 1, 1))

    def forward(s, x):
        return s.net(x)


class Gan:
    def __init__(s, device, lr=1e-4, crop=128):
        s.d = Critic().to(device)
        s.opt = torch.optim.Adam(s.d.parameters(), lr=lr, betas=(0.5, 0.99))
        s.crop = crop
        s.win = (0, 0)

    def pick(s, H, W):
        """One window for this step: the critic's two passes and the denoiser's see the same place."""
        c = min(s.crop, H, W)
        s.win = (random.randint(0, H - c), random.randint(0, W - c), c)

    def view(s, frames):
        y, x, c = s.win
        return frames.flatten(0, 1)[..., y : y + c, x : x + c].float()

    def g_loss(s, outs):
        """What the denoiser minimises: the critic's score of its output, negated."""
        return -s.d(s.view(outs)).mean()

    def d_step(s, outs, clean):
        """One step of the critic (hinge loss). Returns its two parts."""
        s.opt.zero_grad(set_to_none=True)
        lr_ = F.relu(1.0 - s.d(s.view(clean))).mean()
        lf_ = F.relu(1.0 + s.d(s.view(outs.detach()))).mean()
        (lr_ + lf_).backward()
        s.opt.step()
        return lr_.item(), lf_.item()
