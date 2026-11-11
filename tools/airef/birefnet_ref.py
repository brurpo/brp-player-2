"""BiRefNet (matting) in plain PyTorch, written from the paper's code (MIT) for
checking our Vulkan runner. Inference only, the configuration the released matting
weights use: Swin v1 backbone run at full and half size (features concatenated),
the three finer levels squeezed into the coarsest, BasicDecBlk with ASPPDeformable,
image tiles fed to every decoder level, gradient attention.

  python birefnet_ref.py weights.safetensors in.png out_dir [size] [t|l]

Writes alpha.png, alpha.npy (sigmoid, at `size`, default 1024) and the
intermediate features as .npy (for comparing layer by layer).
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from safetensors.torch import load_file
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SWIN = {'t': dict(embed=96, depths=[2, 2, 6, 2], heads=[3, 6, 12, 24], win=7),
        'l': dict(embed=192, depths=[2, 2, 18, 2], heads=[6, 12, 24, 48], win=12)}


def win_part(x, w):
    B, H, W, C = x.shape
    return x.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)


def win_rev(x, w, H, W):
    B = x.shape[0] // ((H // w) * (W // w))
    return x.view(B, H // w, W // w, w, w, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class Attn(nn.Module):
    def __init__(s, dim, heads, win):
        super().__init__()
        s.heads, s.win = heads, win
        s.scale = (dim // heads) ** -0.5
        s.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win - 1) ** 2, heads))
        c = torch.stack(torch.meshgrid(torch.arange(win), torch.arange(win), indexing='ij')).flatten(1)
        r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (win - 1)
        s.register_buffer('relative_position_index', r[:, :, 0] * (2 * win - 1) + r[:, :, 1])
        s.qkv = nn.Linear(dim, dim * 3)
        s.proj = nn.Linear(dim, dim)

    def forward(s, x, mask):
        B, N, C = x.shape
        q, k, v = s.qkv(x).reshape(B, N, 3, s.heads, C // s.heads).permute(2, 0, 3, 1, 4)
        a = (q * s.scale) @ k.transpose(-2, -1)
        bias = s.relative_position_bias_table[s.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        a = a + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            a = (a.view(B // nW, nW, s.heads, N, N) + mask[None, :, None]).view(-1, s.heads, N, N)
        a = a.softmax(-1)
        return s.proj((a @ v).transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.fc1, s.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(s, x):
        return s.fc2(F.gelu(s.fc1(x)))


class Block(nn.Module):
    def __init__(s, dim, heads, win, shift):
        super().__init__()
        s.win, s.shift = win, shift
        s.norm1, s.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        s.attn, s.mlp = Attn(dim, heads, win), Mlp(dim)

    def forward(s, x, H, W, mask):
        B, L, C = x.shape
        sc = x
        x = s.norm1(x).view(B, H, W, C)
        pr, pb = (s.win - W % s.win) % s.win, (s.win - H % s.win) % s.win
        x = F.pad(x, (0, 0, 0, pr, 0, pb))
        Hp, Wp = x.shape[1:3]
        if s.shift:
            x = torch.roll(x, (-s.shift, -s.shift), (1, 2))
        x = win_rev(s.attn(win_part(x, s.win), mask if s.shift else None), s.win, Hp, Wp)
        if s.shift:
            x = torch.roll(x, (s.shift, s.shift), (1, 2))
        x = x[:, :H, :W].reshape(B, H * W, C)
        x = sc + x
        return x + s.mlp(s.norm2(x))


class Merge(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.norm, s.reduction = nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(s, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1).view(B, -1, 4 * C)
        return s.reduction(s.norm(x))


class Layer(nn.Module):
    def __init__(s, dim, depth, heads, win, down):
        super().__init__()
        s.win = win
        s.blocks = nn.ModuleList([Block(dim, heads, win, 0 if i % 2 == 0 else win // 2) for i in range(depth)])
        s.downsample = Merge(dim) if down else None

    def mask(s, H, W, dev):
        w, sh = s.win, s.win // 2
        Hp, Wp = -(-H // w) * w, -(-W // w) * w
        m = torch.zeros(1, Hp, Wp, 1, device=dev)
        cnt = 0
        for h in (slice(0, -w), slice(-w, -sh), slice(-sh, None)):
            for ww in (slice(0, -w), slice(-w, -sh), slice(-sh, None)):
                m[:, h, ww] = cnt
                cnt += 1
        mw = win_part(m, w).squeeze(-1)
        a = mw.unsqueeze(1) - mw.unsqueeze(2)
        return a.masked_fill(a != 0, -100.0).masked_fill(a == 0, 0.0)

    def forward(s, x, H, W):
        mask = s.mask(H, W, x.device)
        for b in s.blocks:
            x = b(x, H, W, mask)
        if s.downsample is None:
            return x, H, W, x, H, W
        return x, H, W, s.downsample(x, H, W), (H + 1) // 2, (W + 1) // 2


class PatchEmbed(nn.Module):
    def __init__(s, dim):
        super().__init__()
        s.proj, s.norm = nn.Conv2d(3, dim, 4, 4), nn.LayerNorm(dim)

    def forward(s, x):
        _, _, H, W = x.shape
        x = F.pad(x, (0, (4 - W % 4) % 4, 0, (4 - H % 4) % 4))
        x = s.proj(x)
        B, C, h, w = x.shape
        return s.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, h, w)


class Swin(nn.Module):
    def __init__(s, embed, depths, heads, win):
        super().__init__()
        s.patch_embed = PatchEmbed(embed)
        s.layers = nn.ModuleList([Layer(embed * 2 ** i, depths[i], heads[i], win, i < 3) for i in range(4)])
        s.dims = [embed * 2 ** i for i in range(4)]
        for i in range(4):
            setattr(s, f'norm{i}', nn.LayerNorm(s.dims[i]))

    def forward(s, x):
        x = s.patch_embed(x)
        Wh, Ww = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        outs = []
        for i, l in enumerate(s.layers):
            xo, H, W, x, Wh, Ww = l(x, Wh, Ww)
            xo = getattr(s, f'norm{i}')(xo)
            outs.append(xo.view(-1, H, W, s.dims[i]).permute(0, 3, 1, 2).contiguous())
        return outs


class DeformConv(nn.Module):
    def __init__(s, cin, cout, k, pad):
        super().__init__()
        s.pad = pad
        s.offset_conv = nn.Conv2d(cin, 2 * k * k, k, 1, pad)
        s.modulator_conv = nn.Conv2d(cin, k * k, k, 1, pad)
        s.regular_conv = nn.Conv2d(cin, cout, k, 1, pad, bias=False)

    def forward(s, x):
        off = s.offset_conv(x)
        mod = 2.0 * torch.sigmoid(s.modulator_conv(x))
        return deform_conv2d(x, off, s.regular_conv.weight, None, padding=s.pad, mask=mod)


class AsppDeformMod(nn.Module):
    def __init__(s, cin, k):
        super().__init__()
        s.atrous_conv, s.bn = DeformConv(cin, 256, k, k // 2), nn.BatchNorm2d(256)

    def forward(s, x):
        return F.relu(s.bn(s.atrous_conv(x)))


class AsppDeform(nn.Module):
    def __init__(s, cin):
        super().__init__()
        s.aspp1 = AsppDeformMod(cin, 1)
        s.aspp_deforms = nn.ModuleList([AsppDeformMod(cin, k) for k in (1, 3, 7)])
        s.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        s.conv1, s.bn1 = nn.Conv2d(256 * 5, cin, 1, bias=False), nn.BatchNorm2d(cin)

    def forward(s, x):
        x1 = s.aspp1(x)
        xs = [m(x) for m in s.aspp_deforms]
        x5 = s.global_avg_pool(x).expand(-1, -1, *x1.shape[2:])
        return F.relu(s.bn1(s.conv1(torch.cat([x1, *xs, x5], 1))))


class DecBlk(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.conv_in, s.bn_in = nn.Conv2d(cin, 64, 3, 1, 1), nn.BatchNorm2d(64)
        s.dec_att = AsppDeform(64)
        s.conv_out, s.bn_out = nn.Conv2d(64, cout, 3, 1, 1), nn.BatchNorm2d(cout)

    def forward(s, x):
        x = F.relu(s.bn_in(s.conv_in(x)))
        return s.bn_out(s.conv_out(s.dec_att(x)))


class Lat(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.conv = nn.Conv2d(c, c, 1)

    def forward(s, x):
        return s.conv(x)


class Simple(nn.Module):
    def __init__(s, cin, cout):
        super().__init__()
        s.conv1, s.conv_out = nn.Conv2d(cin, 64, 3, 1, 1), nn.Conv2d(64, cout, 3, 1, 1)

    def forward(s, x):
        return s.conv_out(s.conv1(x))


def gdt(cin):
    return nn.Sequential(nn.Conv2d(cin, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU())


def tiles(x, ref):
    """b c (hg h) (wg w) -> b (c hg wg) h w: the image cut in tiles the size of `ref`."""
    B, C, H, W = x.shape
    h, w = ref.shape[2:]
    hg, wg = H // h, W // w
    return x.view(B, C, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C * hg * wg, h, w)


def up(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class Decoder(nn.Module):
    def __init__(s, ch):
        super().__init__()
        s.ipt_blk5 = Simple(3 * 2 ** 10, ch[0] // 8)
        s.ipt_blk4 = Simple(3 * 2 ** 8, ch[0] // 8)
        s.ipt_blk3 = Simple(3 * 2 ** 6, ch[1] // 8)
        s.ipt_blk2 = Simple(3 * 2 ** 4, ch[2] // 8)
        s.ipt_blk1 = Simple(3, ch[3] // 8)
        s.decoder_block4 = DecBlk(ch[0] + ch[0] // 8, ch[1])
        s.decoder_block3 = DecBlk(ch[1] + ch[0] // 8, ch[2])
        s.decoder_block2 = DecBlk(ch[2] + ch[1] // 8, ch[3])
        s.decoder_block1 = DecBlk(ch[3] + ch[2] // 8, ch[3] // 2)
        s.conv_out1 = nn.Sequential(nn.Conv2d(ch[3] // 2 + ch[3] // 8, 1, 1))
        s.lateral_block4, s.lateral_block3, s.lateral_block2 = Lat(ch[1]), Lat(ch[2]), Lat(ch[3])
        # Training-only heads (loaded, unused).
        s.conv_ms_spvn_4, s.conv_ms_spvn_3, s.conv_ms_spvn_2 = nn.Conv2d(ch[1], 1, 1), nn.Conv2d(ch[2], 1, 1), nn.Conv2d(ch[3], 1, 1)
        s.gdt_convs_4, s.gdt_convs_3, s.gdt_convs_2 = gdt(ch[1]), gdt(ch[2]), gdt(ch[3])
        for n in ('pred', 'attn'):
            for i in (4, 3, 2):
                setattr(s, f'gdt_convs_{n}_{i}', nn.Sequential(nn.Conv2d(16, 1, 1)))

    def forward(s, x, x1, x2, x3, x4, save):
        x4 = torch.cat([x4, s.ipt_blk5(up(tiles(x, x4), x4.shape[2:]))], 1)
        p4 = s.decoder_block4(x4)
        p4 = p4 * s.gdt_convs_attn_4(s.gdt_convs_4(p4)).sigmoid()
        save('p4', p4)
        p3 = up(p4, x3.shape[2:]) + s.lateral_block4(x3)
        p3 = torch.cat([p3, s.ipt_blk4(up(tiles(x, p3), x3.shape[2:]))], 1)
        p3 = s.decoder_block3(p3)
        p3 = p3 * s.gdt_convs_attn_3(s.gdt_convs_3(p3)).sigmoid()
        save('p3', p3)
        p2 = up(p3, x2.shape[2:]) + s.lateral_block3(x2)
        p2 = torch.cat([p2, s.ipt_blk3(up(tiles(x, p2), x2.shape[2:]))], 1)
        p2 = s.decoder_block2(p2)
        p2 = p2 * s.gdt_convs_attn_2(s.gdt_convs_2(p2)).sigmoid()
        save('p2', p2)
        p1 = up(p2, x1.shape[2:]) + s.lateral_block2(x1)
        p1 = torch.cat([p1, s.ipt_blk2(up(tiles(x, p1), x1.shape[2:]))], 1)
        p1 = s.decoder_block1(p1)
        save('p1', p1)
        p1 = up(p1, x.shape[2:])
        p1 = torch.cat([p1, s.ipt_blk1(up(tiles(x, p1), x.shape[2:]))], 1)
        return s.conv_out1(p1)


class BiRefNet(nn.Module):
    def __init__(s, swin):
        super().__init__()
        s.bb = Swin(**swin)
        ch = [d * 2 for d in s.bb.dims[::-1]]
        s.squeeze_module = nn.Sequential(DecBlk(ch[0] + sum(ch[1:]), ch[0]))
        s.decoder = Decoder(ch)

    def forward(s, x, save=lambda n, t: None):
        H, W = x.shape[2:]
        f = s.bb(x)
        g = s.bb(up(x, (H // 2, W // 2)))
        f = [torch.cat([a, up(b, a.shape[2:])], 1) for a, b in zip(f, g)]
        for i, t in enumerate(f):
            save(f'x{i + 1}', t)
        x1, x2, x3, x4 = f
        x4 = torch.cat([up(x1, x4.shape[2:]), up(x2, x4.shape[2:]), up(x3, x4.shape[2:]), x4], 1)
        x4 = s.squeeze_module(x4)
        save('sq', x4)
        return s.decoder(x, x1, x2, x3, x4, save)


def main():
    wpath, img, out = sys.argv[1:4]
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    kind = sys.argv[5] if len(sys.argv) > 5 else 't'
    os.makedirs(out, exist_ok=True)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in load_file(wpath).items()}
    net = BiRefNet(SWIN[kind])
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith('relative_position_index')]
    unexpected = [k for k in unexpected if not k.endswith('num_batches_tracked')]
    assert not missing and not unexpected, (missing, unexpected)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.eval().to(dev)
    im = Image.open(img).convert('RGB').resize((size, size), Image.BILINEAR)
    im.save(os.path.join(out, 'resized.png'))
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(dev)
    np.save(os.path.join(out, 'input.npy'), x[0].cpu().numpy())
    save = lambda n, t: np.save(os.path.join(out, n + '.npy'), t[0].float().cpu().numpy())
    with torch.no_grad():
        y = net(x, save).sigmoid()[0, 0].cpu().numpy()
    np.save(os.path.join(out, 'alpha.npy'), y)
    Image.fromarray((y * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(os.path.join(out, 'alpha.png'))
    print('alpha', y.shape, 'mean', y.mean(), 'min', y.min(), 'max', y.max())


if __name__ == '__main__':
    main()
