import torch
import torch.nn as nn

from model import VLP, ResidualFeatureAdapter

# TimeSformer model configs
TIMESFORMER_CONFIGS = {
    "vits": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vitb": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "vitl": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}


class _DummyBackbone(nn.Module):
    """Minimal stub to satisfy VLP.__init__ before we replace self.visual."""
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
    def forward(self, x):
        return x


class VLPWithTimeSformer(VLP):
    """VLP variant using SurgCLIP-style TimeSformer (divided space-time attention)
    as the vision encoder. Supports ViT-B and ViT-L backbones."""

    def __init__(self, **kwargs):
        # Determine model size: env var TIMESFORMER_SIZE > vision_backbone kwarg > default
        import os
        backbone = os.environ.get("TIMESFORMER_SIZE", "").strip().lower()
        if not backbone:
            backbone = str(kwargs.get("vision_backbone", "vitl")).lower()
        if backbone not in TIMESFORMER_CONFIGS:
            backbone = "vitl"
        ts_cfg = TIMESFORMER_CONFIGS[backbone]
        self._backbone_name = backbone
        ts_dim = ts_cfg["embed_dim"]

        # Temporarily replace build_visual_backbone to avoid loading heavy weights
        import model as model_module
        _orig_build = model_module.build_visual_backbone
        model_module.build_visual_backbone = lambda name, weights: _DummyBackbone(ts_dim)

        super().__init__(**kwargs)

        # Restore original builder
        model_module.build_visual_backbone = _orig_build

        # Replace visual with actual TimeSformer
        self._build_timesformer(ts_cfg, kwargs.get("num_frames", 8))

        # TimeSformer already does space-time modeling → no frame_pool
        self.frame_pool = None

        # Recreate heads for TimeSformer's output dim
        self.visual_dim = ts_dim
        self.frame_token_dim = ts_dim

        embed_dim = self.embed_dim
        self.video_adapter = ResidualFeatureAdapter(ts_dim)
        self.video_projection = nn.Linear(ts_dim, embed_dim, bias=False)
        self.frame_local_projection = nn.Linear(ts_dim, embed_dim, bias=False)
        self.selection_frame_key_projection = nn.Linear(ts_dim, embed_dim, bias=False)

    def _build_timesformer(self, cfg, num_frames):
        from surgclip.surgclip.models.timesformer.timesformer import TimeSformer

        self.visual = TimeSformer(
            img_size=224,
            patch_size=16,
            num_classes=0,
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            num_frames=num_frames,
            attention_type="divided_space_time",
            gradient_checkpointing=self.encoder_gradient_checkpointing,
        )

        # Try to load EndoSSL weights if available
        from pathlib import Path
        ckpt_name = f"endossl_{self._backbone_name}.pth"
        ckpt_path = Path(__file__).parent / ckpt_name
        if ckpt_path.is_file():
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            msg = self.visual.load_state_dict(state_dict, strict=False)
            print(f"[VLPWithTimeSformer] Loaded {ckpt_name}: "
                  f"{len(state_dict)} spatial keys matched, "
                  f"{len(msg.missing_keys)} temporal keys randomly initialized")
        else:
            print(f"[VLPWithTimeSformer] {ckpt_name} not found, using random init")

    def _encode_image_tokens(self, image: torch.Tensor):
        # image shape: [B, T, C, H, W] (from PretrainDataset)
        # TimeSformer expects [B, C, T, H, W]
        video = image.permute(0, 2, 1, 3, 4)

        # all_tokens: [B, T, N+1, D]; pooled: [B, T, D]
        all_tokens, pooled = self.visual(video)

        # Global: average per-frame features
        video_global_hidden = pooled.mean(dim=1)
        # Frame tokens for selection
        frame_tokens = pooled

        return video_global_hidden, frame_tokens

    def freeze_encoders_train_projections(self):
        # 1. Freeze all visual parameters
        for p in self.visual.parameters():
            p.requires_grad = False

        for p in self.text.backbone.parameters():
            p.requires_grad = False

        # 2. Temporal params: always trainable (EndoSSL has no temporal weights)
        for blk in self.visual.model.blocks:
            for name, p in blk.named_parameters():
                if 'temporal' in name:
                    p.requires_grad = True

        # 3. Unfreeze last N blocks (spatial params only)
        if self.train_encoder_base_layers:
            num_blocks = len(self.visual.model.blocks)
            n = int(self.train_encoder_num_stages)
            if n <= 0:
                n = num_blocks
            n = min(n, num_blocks)
            for blk in self.visual.model.blocks[-n:]:
                for name, p in blk.named_parameters():
                    if 'temporal' not in name:
                        p.requires_grad = True

            # 4. Text: unfreeze last 2 layers
            for layer in self.text.backbone.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # 4. Always train heads
        for p in self.video_adapter.parameters():
            p.requires_grad = True
        for p in self.text_adapter.parameters():
            p.requires_grad = True
        for p in self.video_projection.parameters():
            p.requires_grad = True
        for p in self.frame_local_projection.parameters():
            p.requires_grad = True
        for p in self.selection_text_query_projection.parameters():
            p.requires_grad = True
        for p in self.selection_frame_key_projection.parameters():
            p.requires_grad = True
        for p in self.text.text_projection.parameters():
            p.requires_grad = True
        self.logit_scale.requires_grad = True

        # 5. Gradient checkpointing is already set in TimeSformer init

        # 6. Text gradient checkpointing
        if self.encoder_gradient_checkpointing:
            self._enable_text_activation_checkpointing()

    def set_frozen_modules_eval(self):
        # Do NOT call self.visual.eval() — TimeSformer uses LayerNorm (no
        # BatchNorm) and keeps gradient_checkpointing active via train mode.
        # Frozen spatial params are controlled via requires_grad, not train mode.
        self.text.backbone.eval()

        # Temporal params: always in train mode
        for blk in self.visual.model.blocks:
            blk.temporal_norm1.train()
            blk.temporal_attn.train()
            if hasattr(blk, 'temporal_fc'):
                blk.temporal_fc.train()

        # Unfrozen blocks: full block back to train (enables DropPath)
        if self.train_encoder_base_layers:
            num_blocks = len(self.visual.model.blocks)
            n = int(self.train_encoder_num_stages)
            if n <= 0:
                n = num_blocks
            n = min(n, num_blocks)
            for blk in self.visual.model.blocks[-n:]:
                blk.train()

            for layer in self.text.backbone.encoder.layer[-2:]:
                layer.train()

        self.video_adapter.train()
        self.text_adapter.train()
