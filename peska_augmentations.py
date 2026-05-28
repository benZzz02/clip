"""
SimCLR-style video augmentations using torchvision.transforms.functional.

Thin wrapper: uses TF.adjust_brightness, TF.hflip, F.interpolate etc.
Same crop/flip applied to all T frames for temporal consistency.
"""

import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class SimCLRVideoAugmentation:
    """Apply SimCLR augmentations to video [T, C, H, W] in [0, 1] range."""

    def __init__(self, size=224, scale=(0.2, 1.0), color_jitter_strength=0.4,
                 grayscale_prob=0.2, blur_sigma=(0.1, 2.0), blur_kernel=23):
        self.size = size
        self.scale = scale
        self.ratio = (3.0 / 4.0, 4.0 / 3.0)
        self.cj = color_jitter_strength
        self.gray_p = grayscale_prob
        self.blur_sigma = blur_sigma
        self.blur_k = blur_kernel
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _crop_params(self, h, w):
        area = h * w
        for _ in range(10):
            target = area * random.uniform(*self.scale)
            ar = torch.exp(torch.empty(1).uniform_(
                torch.log(torch.tensor(self.ratio[0])),
                torch.log(torch.tensor(self.ratio[1]))
            )).item()
            nw, nh = int(round((target * ar) ** 0.5)), int(round((target / ar) ** 0.5))
            if 0 < nw <= w and 0 < nh <= h:
                return random.randint(0, h - nh), random.randint(0, w - nw), nh, nw
        # Fallback center crop
        nw, nh = (w, int(round(w / min(self.ratio)))) if w / h < min(self.ratio) else \
                 (int(round(h * max(self.ratio))), h) if w / h > max(self.ratio) else (w, h)
        return (h - nh) // 2, (w - nw) // 2, nh, nw

    def _color_jitter(self, frame):
        b = random.uniform(max(0, 1 - self.cj), 1 + self.cj)
        c = random.uniform(max(0, 1 - self.cj), 1 + self.cj)
        frame = TF.adjust_brightness(frame, b)
        frame = TF.adjust_contrast(frame, c)
        if frame.shape[0] == 3:
            frame = TF.adjust_saturation(frame, random.uniform(max(0, 1 - self.cj), 1 + self.cj))
            frame = TF.adjust_hue(frame, random.uniform(-self.cj * 0.5, self.cj * 0.5))
        return frame

    def _blur(self, frame):
        if random.random() < 0.5:
            return frame
        k = self.blur_k + 1 if self.blur_k % 2 == 0 else self.blur_k
        return F.avg_pool2d(frame.unsqueeze(0), k, 1, k // 2).squeeze(0)

    def __call__(self, video):
        T, C, H, W = video.shape
        i, j, h, w = self._crop_params(H, W)
        do_flip = random.random() < 0.5

        frames = []
        for t in range(T):
            f = video[t][:, i:i + h, j:j + w]
            f = F.interpolate(f.unsqueeze(0), (self.size, self.size),
                              mode='bilinear', align_corners=False).squeeze(0)
            if do_flip:
                f = TF.hflip(f)
            f = self._color_jitter(f)
            if random.random() < self.gray_p and f.shape[0] == 3:
                f = TF.rgb_to_grayscale(f, 3)
            f = self._blur(f)
            frames.append(f)

        video = torch.stack(frames)
        return (video - self.mean.to(video.device)) / self.std.to(video.device)
