"""Temporal heat-map ball detector (docs/BALL_RESEARCH_PLAN.md, section 2).

Input: K=3 consecutive frames + a median background frame of the rally, RGB, 768x432 (12 channels). The background
idea is TrackNetV3's: static objects (logos, a spare ball by the fence) are identical in the frames and the
background and fade out; the match ball moves. Multi-frame in / multi-frame out follows TrackNetV2/V3 and WASB.
Network: U-Net with a ResNet-34 encoder (ImageNet weights; the first conv is widened to 12 channels by repeating its
weights / 4) plus a full-resolution stem branch, so a 5-8 px ball is not lost to striding. Output: K heat-map logits
(one per input frame) at full resolution. Loss: penalty-reduced focal loss (CenterNet, Zhou et al. 2019) on Gaussian
targets, masked where a frame has no label.
Pure torch + torchvision; no other dependencies (runs on Kaggle and on the Mac).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

W, H, K, SIGMA = 768, 432, 3, 2.0


def _cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.c = nn.Sequential(_cbr(i + skip, o), _cbr(o, o))

    def forward(self, x, s):
        x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return self.c(torch.cat([x, s], 1))


class BallTrackNet(nn.Module):
    def __init__(self, k: int = K, pretrained: bool = True):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        cin = 3 * (k + 1)
        conv1 = nn.Conv2d(cin, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(r.conv1.weight.repeat(1, k + 1, 1, 1) / (k + 1))
        self.stem0 = nn.Sequential(_cbr(cin, 32), _cbr(32, 32))            # full resolution
        self.enc1 = nn.Sequential(conv1, r.bn1, r.relu)                     # 1/2, 64
        self.enc2 = nn.Sequential(r.maxpool, r.layer1)                      # 1/4, 64
        self.enc3, self.enc4, self.enc5 = r.layer2, r.layer3, r.layer4      # 1/8 128, 1/16 256, 1/32 512
        self.up4, self.up3 = Up(512, 256, 256), Up(256, 128, 128)
        self.up2, self.up1 = Up(128, 64, 64), Up(64, 64, 48)
        self.up0 = Up(48, 32, 32)
        self.head = nn.Conv2d(32, k, 1)
        nn.init.constant_(self.head.bias, -4.6)  # start at p ~ 0.01 (focal-loss practice)

    def forward(self, x):
        s0 = self.stem0(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d = self.up4(e5, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        d = self.up0(d, s0)
        return self.head(d)


def focal_loss(logits, target, mask, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss. target: Gaussian heat-maps (peak 1 at the ball), mask: (B, K) 1 where
    the frame is labelled (a 'no ball' frame has an all-zero target and still counts)."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float()
    neg = 1 - pos
    pos_l = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_l = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    m = mask[:, :, None, None]
    n_pos = (pos * m).sum().clamp(min=1.0)
    return ((pos_l + neg_l) * m).sum() / n_pos


def gaussian(x, y, w: int = W, h: int = H, sigma: float = SIGMA):
    yy = torch.arange(h, dtype=torch.float32)[:, None]
    xx = torch.arange(w, dtype=torch.float32)[None, :]
    g = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    g[int(round(min(max(y, 0), h - 1))), int(round(min(max(x, 0), w - 1)))] = 1.0
    return g


def peaks(hm: torch.Tensor, thr: float = 0.5):
    """Sub-pixel ball position from one heat-map (probabilities) or None: weighted centre of the blob around the max."""
    m = float(hm.max())
    if m < thr:
        return None, m
    i = int(hm.argmax())
    y, x = divmod(i, hm.shape[-1])
    y0, y1, x0, x1 = max(0, y - 3), y + 4, max(0, x - 3), x + 4
    patch = hm[y0:y1, x0:x1]
    yy, xx = torch.meshgrid(torch.arange(y0, y0 + patch.shape[0]), torch.arange(x0, x0 + patch.shape[1]), indexing="ij")
    w = patch.clamp(min=0)
    return (float((xx * w).sum() / w.sum()), float((yy * w).sum() / w.sum())), m
