import torch
import torch.nn as nn

from model import VLP, ResidualFeatureAdapter


class VLPWithTimeSformer(VLP):
    """VLP variant that replaces the EndoSSL ViT-L + temporal pooling
    with a SurgCLIP-style TimeSformer (divided space-time attention)
    initialized from EndoSSL spatial weights."""

    def __init__(self, **kwargs):
        # Let parent set up text encoder, projections, adapters etc.
        # using vision_backbone="vitl" so visual_dim is 1024
        super().__init__(**kwargs)

        # Replace visual with TimeSformer
        self._build_timesformer(kwargs.get("num_frames", 8))

        # TimeSformer already does space-time modeling → no frame_pool
        self.frame_pool = None

        # TimeSformer outputs 1024-dim → recreate heads for this dim
        ts_dim = 1024
        self.visual_dim = ts_dim
        self.frame_token_dim = ts_dim

        embed_dim = self.embed_dim
        self.video_adapter = ResidualFeatureAdapter(ts_dim)
        self.video_projection = nn.Linear(ts_dim, embed_dim, bias=False)
        self.frame_local_projection = nn.Linear(ts_dim, embed_dim, bias=False)
        self.selection_frame_key_projection = nn.Linear(ts_dim, embed_dim, bias=False)

    def _build_timesformer(self, num_frames):
        from surgclip.surgclip.models.timesformer.timesformer import TimeSformer

        self.visual = TimeSformer(
            img_size=224,
            patch_size=16,
            num_classes=0,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            num_frames=num_frames,
            attention_type="divided_space_time",
            gradient_checkpointing=self.encoder_gradient_checkpointing,
        )

        # Load EndoSSL spatial weights; temporal params init randomly
        from pathlib import Path
        ckpt_path = Path(__file__).parent / "endossl_vitl.pth"
        if ckpt_path.is_file():
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            msg = self.visual.load_state_dict(state_dict, strict=False)
            print(f"[VLPWithTimeSformer] Loaded EndoSSL weights: "
                  f"{len(state_dict)} spatial keys matched, "
                  f"{len(msg.missing_keys)} temporal keys randomly initialized")
        else:
            print(f"[VLPWithTimeSformer] WARNING: {ckpt_path} not found, using random init")

    def _encode_image_tokens(self, image: torch.Tensor):
        # image shape: [B, T, C, H, W] (from PretrainDataset)
        # TimeSformer expects [B, C, T, H, W]
        video = image.permute(0, 2, 1, 3, 4)

        # all_tokens: [B, T, N+1, 1024]; pooled: [B, T, 1024]
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

        # 2. Unfreeze last N blocks (spatial + temporal params)
        if self.train_encoder_base_layers:
            num_blocks = len(self.visual.model.blocks)
            n = int(self.train_encoder_num_stages)
            if n <= 0:
                n = num_blocks
            n = min(n, num_blocks)
            for blk in self.visual.model.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

            # 3. Text: unfreeze last 2 layers
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
        # Frozen parts to eval mode
        self.visual.eval()
        self.text.backbone.eval()

        # Unfrozen blocks back to train
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
