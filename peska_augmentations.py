"""
SimCLR-style video augmentations with frame-consistent spatial transforms.

Key design: the SAME random crop region and horizontal flip are applied
to ALL T frames in a clip, preserving temporal consistency. Color jitter
is applied independently per frame (following SimCLR practice).

This matches PeskaVLP's approach in codes/datasets/transform.py.
"""

import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class SimCLRVideoAugmentation:
    """
    Apply SimCLR-style spatial augmentations to a video clip.

    Video input shape: [T, C, H, W] in range [0, 1].
    The same RandomResizedCrop region and HorizontalFlip decision are used
    for all T frames. ColorJitter, Grayscale, and GaussianBlur are applied
    per-frame (as in the original SimCLR).

    Augmentation pipeline:
        1. RandomResizedCrop(size)      - same crop across all frames
        2. RandomHorizontalFlip(p=0.5)  - same flip across all frames
        3. ColorJitter(brightness, contrast, saturation, hue) - per-frame
        4. RandomGrayscale(p=0.2)       - per-frame
        5. GaussianBlur(kernel, sigma)  - per-frame
        6. Normalize(mean, std)         - per-frame
    """

    def __init__(
        self,
        size=224,
        scale=(0.2, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        color_jitter_strength=0.4,
        grayscale_prob=0.2,
        gaussian_blur_sigma=(0.1, 2.0),
        gaussian_kernel_size=23,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.cj_strength = color_jitter_strength
        self.grayscale_prob = grayscale_prob
        self.blur_sigma = gaussian_blur_sigma
        self.blur_kernel = gaussian_kernel_size

        self.register_mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    @staticmethod
    def _get_crop_params(img_h, img_w, scale, ratio):
        """Get parameters for RandomResizedCrop (same as torchvision)."""
        area = img_h * img_w
        log_ratio = (torch.log(torch.tensor(ratio[0])),
                     torch.log(torch.tensor(ratio[1])))

        for _ in range(10):
            target_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round((target_area * aspect_ratio) ** 0.5))
            h = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < w <= img_w and 0 < h <= img_h:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                return i, j, h, w

        # Fallback: center crop
        in_ratio = img_w / img_h
        if in_ratio < min(ratio):
            w = img_w
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img_h
            w = int(round(h * max(ratio)))
        else:
            w = img_w
            h = img_h
        i = (img_h - h) // 2
        j = (img_w - w) // 2
        return i, j, h, w

    def _apply_color_jitter(self, frame):
        """Apply ColorJitter to a SINGLE frame [C, H, W]."""
        brightness_factor = random.uniform(
            max(0, 1 - self.cj_strength), 1 + self.cj_strength
        )
        contrast_factor = random.uniform(
            max(0, 1 - self.cj_strength), 1 + self.cj_strength
        )
        saturation_factor = random.uniform(
            max(0, 1 - self.cj_strength), 1 + self.cj_strength
        )
        hue_factor = random.uniform(
            -self.cj_strength * 0.5, self.cj_strength * 0.5
        )

        frame = TF.adjust_brightness(frame, brightness_factor)
        frame = TF.adjust_contrast(frame, contrast_factor)

        if frame.shape[0] == 3:
            frame = TF.adjust_saturation(frame, saturation_factor)
            frame = TF.adjust_hue(frame, hue_factor)

        return frame

    def _apply_gaussian_blur(self, frame):
        """Apply Gaussian blur to a SINGLE frame [C, H, W] via avg pooling."""
        if random.random() < 0.5:
            return frame

        sigma = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1

        padding = kernel_size // 2
        frame = frame.unsqueeze(0)
        frame = F.avg_pool2d(frame, kernel_size=kernel_size,
                             stride=1, padding=padding)
        return frame.squeeze(0)

    def __call__(self, video):
        """
        Args:
            video: torch.Tensor of shape [T, C, H, W] in range [0, 1]

        Returns:
            torch.Tensor of shape [T, C, H, W], augmented and normalized
        """
        T, C, H, W = video.shape
        device = video.device

        # 1. Sample crop parameters ONCE for all frames
        i, j, h, w = self._get_crop_params(H, W, self.scale, self.ratio)

        # 2. Sample horizontal flip ONCE for all frames
        do_flip = random.random() < 0.5

        # 3. Apply identical spatial transforms to all frames
        augmented = []
        for t in range(T):
            frame = video[t]

            # RandomResizedCrop
            frame = frame[:, i:i + h, j:j + w]
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(self.size, self.size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

            # RandomHorizontalFlip
            if do_flip:
                frame = TF.hflip(frame)

            # ColorJitter (per-frame)
            frame = self._apply_color_jitter(frame)

            # RandomGrayscale (per-frame)
            if random.random() < self.grayscale_prob and frame.shape[0] == 3:
                frame = TF.rgb_to_grayscale(frame, num_output_channels=3)

            # GaussianBlur (per-frame)
            frame = self._apply_gaussian_blur(frame)

            augmented.append(frame)

        video = torch.stack(augmented, dim=0)

        # 4. Normalize
        mean = self.register_mean.to(device)
        std = self.register_std.to(device)
        video = (video - mean) / std

        return video
