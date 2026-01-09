# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Crack segmentation enhancement modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class DilatedContextBlock(nn.Module):
    """D-LinkNet-style dilated context block with residual connection."""

    def __init__(self, c1: int, c2: int | None = None, e: float = 0.5):
        super().__init__()
        c2 = c1 if c2 is None else c2
        cr = max(1, int(c2 * e))
        self.c1 = c1
        self.c2 = c2
        self.cv1 = Conv(c1, cr, k=1)
        self.cv2 = Conv(cr, cr, k=3, p=1, d=1)
        self.cv3 = Conv(cr, cr, k=3, p=2, d=2)
        self.cv4 = Conv(cr, cr, k=3, p=4, d=4)
        self.cv5 = Conv(cr * 3, c2, k=1)
        self.act = nn.SiLU()
        self.proj = Conv(c1, c2, k=1, act=False) if c1 != c2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            assert x.shape[1] == self.c1, f"Expected {self.c1} channels, got {x.shape[1]}"
        y = self.cv1(x)
        y1 = self.cv2(y)
        y2 = self.cv3(y)
        y3 = self.cv4(y)
        y = self.cv5(torch.cat((y1, y2, y3), 1))
        res = x if self.proj is None else self.proj(x)
        return self.act(y + res)


class StripPooling2D(nn.Module):
    """Strip pooling block for anisotropic context aggregation."""

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c1 = c1
        self.c2 = c2
        self.conv_h = Conv(c1, c1, k=(1, 3), p=(0, 1))
        self.conv_w = Conv(c1, c1, k=(3, 1), p=(1, 0))
        self.fuse = Conv(c1 * 2, c2, k=1)
        self.act = nn.SiLU()
        self.proj = Conv(c1, c2, k=1, act=False) if c1 != c2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            assert x.shape[1] == self.c1, f"Expected {self.c1} channels, got {x.shape[1]}"
        _, _, h, w = x.shape
        pool_h = x.mean(dim=2, keepdim=True)
        pool_w = x.mean(dim=3, keepdim=True)
        h_feat = self.conv_h(pool_h)
        w_feat = self.conv_w(pool_w)
        h_feat = F.interpolate(h_feat, size=(h, w), mode="bilinear", align_corners=False)
        w_feat = F.interpolate(w_feat, size=(h, w), mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat((h_feat, w_feat), 1))
        res = x if self.proj is None else self.proj(x)
        return res + self.act(fused)


class SSF2D(nn.Module):
    """Spatial-semantic fusion for high/low resolution features."""

    def __init__(self, c_s: int, c_m: int):
        super().__init__()
        self.c_s = c_s
        self.c_m = c_m
        self.proj = Conv(c_m, c_s, k=1)
        self.gate = Conv(c_s, c_s, k=1, act=False)
        self.refine = Conv(c_s, c_s, k=3, p=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x_s, x_m = x
        else:
            raise TypeError("SSF2D expects a list or tuple of [spatial, semantic] tensors")
        if not torch.jit.is_scripting():
            assert x_s.shape[1] == self.c_s, f"Expected spatial {self.c_s} channels, got {x_s.shape[1]}"
            assert x_m.shape[1] == self.c_m, f"Expected semantic {self.c_m} channels, got {x_m.shape[1]}"
        _, _, h, w = x_s.shape
        x_m = F.interpolate(x_m, size=(h, w), mode="bilinear", align_corners=False)
        m_proj = self.proj(x_m)
        gate = torch.sigmoid(self.gate(m_proj))
        y = x_s * (1 + gate) + self.refine(m_proj)
        return self.act(y)


class CVF2D(nn.Module):
    """Context-view fusion for same-resolution features."""

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c1 = c1
        self.c2 = c2
        self.gate_a = Conv(c2, c1, k=1, act=False)
        self.gate_b = Conv(c1, c1, k=1, act=False)
        self.refine = Conv(c1, c1, k=3, p=1)
        self.act = nn.SiLU()
        self.proj = Conv(c2, c1, k=1, act=False) if c1 != c2 else None

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x_a, x_b = x
        else:
            raise TypeError("CVF2D expects a list or tuple of [view_a, view_b] tensors")
        if not torch.jit.is_scripting():
            assert x_a.shape[1] == self.c1, f"Expected view_a {self.c1} channels, got {x_a.shape[1]}"
            assert x_b.shape[1] == self.c2, f"Expected view_b {self.c2} channels, got {x_b.shape[1]}"
        if self.proj is not None:
            x_b = self.proj(x_b)
        g_a = torch.sigmoid(self.gate_a(x_b))
        g_b = torch.sigmoid(self.gate_b(x_a))
        y = x_a * (1 + g_a) + x_b * (1 + g_b)
        y = self.refine(y) + x_a
        return self.act(y)


class CrackEnhanceBlock(nn.Module):
    """Wrapper block to apply optional dilated context and strip pooling."""

    def __init__(
        self, c1: int, c2: int | None = None, use_dilated: bool = True, use_strip: bool = True, e: float = 0.5
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.c1 = c1
        self.c2 = c2
        self.use_dilated = use_dilated
        self.use_strip = use_strip
        self.dilated = DilatedContextBlock(c1, c2, e=e) if use_dilated else nn.Identity()
        self.strip = StripPooling2D(c2, c2) if use_strip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            assert x.shape[1] == self.c1, f"Expected {self.c1} channels, got {x.shape[1]}"
        x = self.dilated(x)
        return self.strip(x)
