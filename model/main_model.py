import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Utility layers (unchanged) ------------------------------------------------

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class DWSConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, dilation=1, act=True, bias=True):
        super().__init__()
        pad      = dilation * (k // 2)
        self.dw  = nn.Conv2d(in_ch, in_ch, k, stride, pad, dilation, groups=in_ch, bias=bias)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class CALayer(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        hidden   = max(ch // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


# -- New: Multi-Scale Depthwise Separable Conv ---------------------------------

class MultiScaleDWS(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # three parallel depthwise branches
        self.d1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1,
                            groups=ch, bias=False)                 # dilation 1
        self.d2 = nn.Conv2d(ch, ch, kernel_size=3, padding=2,
                            dilation=2, groups=ch, bias=False)     # dilation 2
        self.d4 = nn.Conv2d(ch, ch, kernel_size=3, padding=4,
                            dilation=4, groups=ch, bias=False)     # dilation 4

        # shared pointwise projection (same cost as baseline pw)
        self.pw  = nn.Conv2d(ch, ch, kernel_size=1)
        self.act = nn.GELU()

        # learnable branch importance — equal start, free to specialise
        self.w = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        w   = torch.softmax(self.w, dim=0)          # (3,) sums to 1
        out = w[0] * self.d1(x) \
            + w[1] * self.d2(x) \
            + w[2] * self.d4(x)
        return self.act(self.pw(out))

    def branch_weights(self):
        """Call after training to inspect which dilation each block prefers."""
        w = torch.softmax(self.w.detach(), dim=0)
        return {'d1': w[0].item(), 'd2': w[1].item(), 'd4': w[2].item()}


# -- Restoration block (one line changed) -------------------------------------

class PlainRestorationBlock(nn.Module):
    def __init__(self, ch, expansion=2):
        super().__init__()
        hidden = ch * expansion

        self.norm1 = LayerNorm2d(ch)
        self.body  = nn.Sequential(
            nn.Conv2d(ch, hidden * 2, kernel_size=1),
            SimpleGate(),
            MultiScaleDWS(hidden),               # ? only change from baseline
            CALayer(hidden),
            nn.Conv2d(hidden, ch, kernel_size=1),
        )
        self.beta  = nn.Parameter(torch.zeros(1, ch, 1, 1))

        self.norm2    = LayerNorm2d(ch)
        ffn_hidden    = ch * 2
        self.ffn      = nn.Sequential(
            nn.Conv2d(ch, ffn_hidden * 2, kernel_size=1),
            SimpleGate(),
            CALayer(ffn_hidden),
            nn.Conv2d(ffn_hidden, ch, kernel_size=1),
        )
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        x = x + self.beta  * self.body(self.norm1(x))
        x = x + self.gamma * self.ffn(self.norm2(x))
        return x


class BlockStack(nn.Module):
    def __init__(self, ch, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PlainRestorationBlock(ch) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


# -- Scale transforms (unchanged) ---------------------------------------------

class SimpleDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.GELU(),
            DWSConv(out_ch, out_ch, k=3),
        )

    def forward(self, x):
        return self.proj(x)


class SimpleUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.GELU(),
            DWSConv(out_ch, out_ch, k=3),
        )

    def forward(self, x):
        return self.proj(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))


class SimpleFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.GELU(),
            DWSConv(ch, ch, k=3),
        )

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=1))


# -- Output head (unchanged) ---------------------------------------------------

class OutputHead(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body  = nn.Sequential(
            DWSConv(ch, ch, k=3),
            nn.Conv2d(ch, 3, kernel_size=3, padding=1),
        )
        self.scale = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.1)

    def forward(self, feat, base, clamp=False):
        y = base + self.body(feat) * self.scale
        return torch.clamp(y, 0, 1) if clamp else y


# -- Main model ----------------------------------------------------------------

class DIVENet_Main(nn.Module):

    def __init__(self, clamp_output=False):
        super().__init__()
        c1, c2, c3 = 20, 40, 80
        self.clamp_output = clamp_output

        self.stem  = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1), nn.GELU(), DWSConv(c1, c1, k=3))

        self.enc1  = BlockStack(c1, 2)
        self.down1 = SimpleDown(c1, c2)
        self.enc2  = BlockStack(c2, 2)
        self.down2 = SimpleDown(c2, c3)
        self.enc3  = BlockStack(c3, 3)

        self.dec3  = BlockStack(c3, 1)
        self.out3  = OutputHead(c3)

        self.up3   = SimpleUp(c3, c2)
        self.skip2 = SimpleFusion(c2)
        self.dec2  = BlockStack(c2, 1)
        self.out2  = OutputHead(c2)

        self.up2   = SimpleUp(c2, c1)
        self.skip1 = SimpleFusion(c1)
        self.dec1  = BlockStack(c1, 1)
        self.out1  = OutputHead(c1)

    @staticmethod
    def _pad(x, m=16):
        _, _, h, w = x.shape
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x, h, w

    @staticmethod
    def _align(src, ref):
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(src, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return src

    def forward(self, x):
        x, h0, w0 = self._pad(x)
        x2 = F.interpolate(x,  scale_factor=0.5, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x2, scale_factor=0.5, mode='bilinear', align_corners=False)

        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        d3 = self.dec3(e3)
        y4 = self.out3(d3, x4, clamp=self.clamp_output)

        u2 = self._align(self.up3(d3), e2)
        d2 = self.dec2(self.skip2(u2, e2))
        y2 = self.out2(d2, x2, clamp=self.clamp_output)

        u1 = self._align(self.up2(d2), e1)
        d1 = self.dec1(self.skip1(u1, e1))
        y1 = self.out1(d1, x, clamp=self.clamp_output)

        th4, tw4 = math.ceil(h0 / 4), math.ceil(w0 / 4)
        th2, tw2 = math.ceil(h0 / 2), math.ceil(w0 / 2)
        return [
            y4[:, :, :th4, :tw4],
            y2[:, :, :th2, :tw2],
            y1[:, :, :h0,  :w0 ],
        ]



