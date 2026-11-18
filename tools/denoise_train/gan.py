"""The adversarial test (train.py --gan-weight): a critic trained beside the denoiser to
tell its output from clean frames, so the denoiser may put plausible fine texture where
the exact one cannot be known (the pixel losses alone answer that with blur).

Without --gan-temporal the critic sees single frames only (crops), nothing of time, and
that invented texture stays the same from frame to frame is left to the steadiness loss
(train.stability). That was not enough: the run `t_hybg2` (--stab-weight 2) put 2.7 times
more fine texture into dark flat areas of real footage, new every frame. With
--gan-temporal the critic sees, beside each frame, how it differs from the frame before
moved along the clean frames' motion (train.stability's own difference, zero where the
motion is unreliable), amplified like the detail: clean video hardly changes there, so
texture that crawls gives the output away. Trained from scratch here, nothing pretrained.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """A patch critic: 8 convolutions down to 1/8, about 45 px of the picture per score
    (texture, not content). Spectral normalisation keeps it from running away."""

    def __init__(s, w=64, cin=6):
        super().__init__()
        sn = nn.utils.spectral_norm
        chans = [(cin, w, 3, 1), (w, w, 4, 2), (w, 2 * w, 3, 1), (2 * w, 2 * w, 4, 2), (2 * w, 4 * w, 3, 1), (4 * w, 4 * w, 4, 2), (4 * w, 4 * w, 3, 1)]
        layers = []
        for i, o, k, st in chans:
            layers += [sn(nn.Conv2d(i, o, k, st, 1)), nn.LeakyReLU(0.2, True)]
        s.net = nn.Sequential(*layers, nn.Conv2d(4 * w, 1, 3, 1, 1))

    def forward(s, x):
        return s.net(x)


def detail(x, gain):
    """The picture and, beside it, its fine detail (the picture minus a 5 px blur of it)
    times `gain`. The model's output is within about a level of the clean frame, and a
    critic with spectral normalisation cannot turn a difference that small into a
    different score (measured: both scored the same to 0.001 for 700 steps); amplified,
    the texture is as large as the picture's own contrasts."""
    blur = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)
    return torch.cat([x, (x - blur) * gain], 1)


class Gan:
    def __init__(s, device, lr=1e-4, crop=128, gain=16.0, temporal=False):
        s.temporal = temporal
        s.d = Critic(cin=9 if temporal else 6).to(device)
        s.gain = gain
        s.opt = torch.optim.Adam(s.d.parameters(), lr=lr, betas=(0.5, 0.99))
        s.crop = crop
        s.win = (0, 0)

    def pick(s, H, W):
        """One window for this step: the critic's two passes and the denoiser's see the same place."""
        c = min(s.crop, H, W)
        s.win = (random.randint(0, H - c), random.randint(0, W - c), c)

    def view(s, frames, change=None):
        """`frames` (B, T, 3, H, W); `change` (B * (T - 1), 3, H, W): frames 1.. minus the
        frame before moved along the motion (--gan-temporal; the first frame of each clip
        has none and is left out)."""
        y, x, c = s.win
        if not s.temporal:
            return detail(frames.flatten(0, 1)[..., y : y + c, x : x + c].float(), s.gain)
        cur = frames[:, 1:].flatten(0, 1)[..., y : y + c, x : x + c].float()
        return torch.cat([detail(cur, s.gain), change[..., y : y + c, x : x + c].float() * s.gain], 1)

    def g_loss(s, outs, change=None):
        """What the denoiser minimises: the critic's score of its output, negated."""
        return -s.d(s.view(outs, change)).mean()

    def d_step(s, outs, clean, change_out=None, change_clean=None):
        """One step of the critic (hinge loss). Returns its two parts."""
        s.opt.zero_grad(set_to_none=True)
        lr_ = F.relu(1.0 - s.d(s.view(clean, change_clean))).mean()
        lf_ = F.relu(1.0 + s.d(s.view(outs.detach(), change_out))).mean()
        (lr_ + lf_).backward()
        s.opt.step()
        return lr_.item(), lf_.item()
