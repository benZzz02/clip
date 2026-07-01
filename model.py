import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import AutoConfig, AutoModel

from surgclip.surgclip.models.timesformer.timesformer import Block
from visual_backbones import build_LemonFM, build_visual_backbone


def _checkpoint_block_forward(module):
    """Wrap a module's forward with activation checkpointing (use_reentrant=False)."""
    if getattr(module, "_vlp_ckpt_wrapped", False):
        return
    original_forward = module.forward

    def ckpt_forward(*args, **kwargs):
        if not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        if kwargs:
            def _run(*a):
                return original_forward(*a, **kwargs)
            return torch_checkpoint(_run, *args, use_reentrant=False)
        return torch_checkpoint(original_forward, *args, use_reentrant=False)

    module.forward = ckpt_forward
    module._vlp_ckpt_wrapped = True


def apply_convnext_gradient_checkpointing(visual, stage_indices=None):
    """Wrap each CNBlock inside specified ConvNeXt stages with activation checkpointing."""
    base_model = visual.model if hasattr(visual, "model") else visual
    if hasattr(base_model, "get_base_model"):
        base_model = base_model.get_base_model()
    features = getattr(base_model, "features", None)
    if features is None:
        return 0
    wrapped = 0
    for i, stage in enumerate(features):
        if stage_indices is not None and i not in stage_indices:
            continue
        for module in stage.modules():
            if module.__class__.__name__ in {"CNBlock", "Block"}:
                _checkpoint_block_forward(module)
                wrapped += 1
    return wrapped


def apply_vit_gradient_checkpointing(visual, num_blocks):
    """Wrap last num_blocks ViT transformer blocks with activation checkpointing."""
    base_model = visual.model if hasattr(visual, "model") else visual
    if hasattr(base_model, "get_base_model"):
        base_model = base_model.get_base_model()
    blocks = getattr(base_model, "blocks", None)
    if blocks is None:
        return 0
    wrapped = 0
    for blk in blocks[-num_blocks:]:
        _checkpoint_block_forward(blk)
        wrapped += 1
    return wrapped


class SurgicBERTaTextEncoder(nn.Module):
    def __init__(self, model_name="marcobombieri/surgicberta", embed_dim=512):
        super().__init__()

        model_name = model_name.strip()
        lower_name = model_name.lower()

        if lower_name.startswith(("random:", "scratch:", "none:")):
            base_name = model_name.split(":", 1)[1].strip()
            if not base_name:
                raise ValueError(
                    "Random text initialization requires a base config name, "
                    "e.g. random:bert-base-uncased"
                )
            print(f"Initializing text backbone from config only: {base_name}")
            config = AutoConfig.from_pretrained(base_name)
            config.add_pooling_layer = False
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                add_pooling_layer=False,
            )

        self.hidden_size = self.backbone.config.hidden_size
        self.text_projection = nn.Linear(self.hidden_size, embed_dim, bias=False)

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def project(self, pooled_hidden):
        return self.text_projection(pooled_hidden)

    def forward(self, input_ids, attention_mask, return_hidden=False):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_hidden = outputs.last_hidden_state
        pooled_hidden = self.mean_pooling(
            token_hidden,
            attention_mask,
        )
        text_features = self.project(pooled_hidden)
        if return_hidden:
            return text_features, token_hidden, pooled_hidden
        return text_features


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=None, bottleneck_ratio=4):
        super().__init__()
        feature_dim = int(feature_dim)
        if bottleneck_dim is None:
            bottleneck_dim = max(1, feature_dim // int(bottleneck_ratio))

        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, int(bottleneck_dim))
        self.act = nn.GELU()
        self.up = nn.Linear(int(bottleneck_dim), feature_dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class TimeSformerStyleTemporalPool(nn.Module):
    """
    Temporal head for frame-level features [B, T, D].

    This keeps the ConvNeXt image encoder, but replaces the old frame attention
    pooling with a lightweight TimeSformer-style transformer head over frame tokens.
    """

    def __init__(
        self,
        frame_dim,
        hidden_dim=768,
        num_frames=8,
        depth=2,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_drop=0.0,
        drop_path=0.1,
        use_cls_token=True,
    ):
        super().__init__()

        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim
        self.num_frames = max(1, int(num_frames))
        self.use_cls_token = bool(use_cls_token)

        self.input_norm = nn.LayerNorm(frame_dim)
        self.input_proj = (
            nn.Identity()
            if frame_dim == hidden_dim
            else nn.Linear(frame_dim, hidden_dim, bias=False)
        )

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, hidden_dim))
            if self.use_cls_token
            else None
        )
        self.time_embed = nn.Parameter(torch.zeros(1, self.num_frames, hidden_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=dropout,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    attention_type="joint_space_time",
                    gradient_checkpointing=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.time_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def _resize_time_embed(self, t):
        if t == self.time_embed.size(1):
            return self.time_embed

        time_embed = self.time_embed.transpose(1, 2)  # [1, D, T0]
        time_embed = F.interpolate(time_embed, size=t, mode="nearest")
        return time_embed.transpose(1, 2)  # [1, T, D]

    def forward(self, frame_features, return_tokens=False):
        if frame_features.ndim != 3:
            raise ValueError(
                f"Expected frame_features shape [B, T, D], got {tuple(frame_features.shape)}"
            )

        b, t, _ = frame_features.shape

        x = self.input_norm(frame_features)
        x = self.input_proj(x)
        x = x + self._resize_time_embed(t)

        if self.cls_token is not None:
            cls = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.pos_drop(x)

        # Reuse TimeSformer blocks as frame-token self-attention blocks.
        # With W=1 and attention_type="joint_space_time", this behaves like a
        # standard transformer over the sequence [cls, frame_1, ..., frame_T].
        for blk in self.blocks:
            x = blk(x, B=b, T=t, W=1)

        x = self.norm(x)

        if self.cls_token is not None:
            frame_tokens = x[:, 1:]
            pooled = 0.5 * x[:, 0] + 0.5 * frame_tokens.mean(dim=1)
        else:
            frame_tokens = x
            pooled = x.mean(dim=1)

        if return_tokens:
            return pooled, frame_tokens
        return pooled


class VLP(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        text_model_name="marcobombieri/surgicberta",
        vision_backbone="convnext_lemonfm",
        vision_pretrained_weights="lemonfm.pth",
        num_frames=4,
        temporal_num_layers=2,
        temporal_num_heads=12,
        temporal_dropout=0.1,
        temporal_hidden_dim=768,
        local_temperature=0.07,
        selection_pooling="similarity",
        level_frame_temperatures=(0.6, 0.9, 1.2),
        train_encoder_base_layers=True,
        encoder_gradient_checkpointing=False,
        train_encoder_num_stages=1,
    ):
        super().__init__()

        self.vision_backbone = str(vision_backbone)
        self.visual = build_visual_backbone(
            name=vision_backbone,
            weights=vision_pretrained_weights,
        )
        self.text = SurgicBERTaTextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
        )

        self.visual_dim = self.visual.output_dim
        self.embed_dim = embed_dim
        self.num_frames = max(1, int(num_frames))
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.local_temperature = float(local_temperature)
        self.selection_pooling = str(selection_pooling).lower()
        self.train_encoder_base_layers = bool(train_encoder_base_layers)
        self.encoder_gradient_checkpointing = bool(encoder_gradient_checkpointing)
        self.train_encoder_num_stages = int(train_encoder_num_stages)

        self.frame_pool = None
        if self.num_frames > 1:
            self.frame_pool = TimeSformerStyleTemporalPool(
                frame_dim=self.visual_dim,
                hidden_dim=self.temporal_hidden_dim,
                num_frames=self.num_frames,
                depth=temporal_num_layers,
                num_heads=temporal_num_heads,
                mlp_ratio=4.0,
                dropout=temporal_dropout,
                attn_drop=0.0,
                drop_path=0.1,
                use_cls_token=True,
            )

        pooled_dim = self.temporal_hidden_dim if self.frame_pool is not None else self.visual_dim
        self.frame_token_dim = pooled_dim
        self.video_adapter = ResidualFeatureAdapter(pooled_dim)
        self.text_adapter = ResidualFeatureAdapter(self.text.hidden_size)
        self.video_projection = nn.Linear(
            pooled_dim,
            embed_dim,
            bias=False,
        )
        self.frame_local_projection = nn.Linear(self.frame_token_dim, embed_dim, bias=False)
        self.selection_text_query_projection = nn.Linear(self.text.hidden_size, embed_dim, bias=False)
        self.selection_frame_key_projection = nn.Linear(self.frame_token_dim, embed_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        nn.init.normal_(self.video_projection.weight, std=pooled_dim ** -0.5)
        nn.init.normal_(self.frame_local_projection.weight, std=embed_dim ** -0.5)
        nn.init.normal_(self.selection_text_query_projection.weight, std=embed_dim ** -0.5)
        nn.init.normal_(self.selection_frame_key_projection.weight, std=embed_dim ** -0.5)

        self.register_buffer(
            "level_frame_temperatures",
            torch.tensor(level_frame_temperatures, dtype=torch.float32),
            persistent=False,
        )

        self.last_frame_weights = None
        self.last_token_weights = None
        self.last_pair_confidence = None
        self.last_pair_weights = None
        self.last_entropy_regularization = None
        self.last_frame_entropy = None
        self.last_token_entropy = None
        self.last_video_gate = None
        self.last_text_gate = None
        self.last_frame_peak = None
        self.last_token_peak = None
        self.last_distill_regularization = None

    def _prepare_image_input(self, image: torch.Tensor):
        if image.ndim == 4:
            image = image.unsqueeze(1)
        elif image.ndim != 5:
            raise ValueError(
                f"Expected image shape [B, C, H, W] or [B, T, C, H, W], got {tuple(image.shape)}"
            )
        return image

    def _get_level_values(self, level_ids, values, default_value, batch_size, device, dtype):
        if level_ids is None:
            return torch.full((batch_size,), float(default_value), device=device, dtype=dtype)

        level_ids = level_ids.to(device=device, dtype=torch.long).clamp(min=0, max=len(values) - 1)
        return values.to(device=device, dtype=dtype)[level_ids]

    def _masked_logsumexp(self, scores, mask, dim):
        if mask is None:
            return torch.logsumexp(scores, dim=dim)
        masked_scores = scores.masked_fill(~mask, -1e4)
        return torch.logsumexp(masked_scores, dim=dim)

    def _masked_max(self, scores, mask, dim):
        if mask is None:
            return scores.max(dim=dim).values
        masked_scores = scores.masked_fill(~mask, -1e4)
        return masked_scores.max(dim=dim).values

    def _masked_mean(self, scores, mask, dim):
        if mask is None:
            return scores.mean(dim=dim)
        masked_scores = scores * mask.to(dtype=scores.dtype)
        denom = mask.sum(dim=dim).clamp(min=1).to(dtype=scores.dtype)
        return masked_scores.sum(dim=dim) / denom

    def _normalize_scores(self, scores, mask=None, topk=None, temperatures=1.0):
        if not torch.is_tensor(temperatures):
            temperatures = torch.full(
                (scores.size(0),),
                float(temperatures),
                device=scores.device,
                dtype=scores.dtype,
            )
        temperatures = temperatures.to(device=scores.device, dtype=scores.dtype).clamp(min=1e-4)
        scaled_scores = scores / temperatures.unsqueeze(-1)

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.to(device=scores.device, dtype=torch.bool)

        scaled_scores = scaled_scores.masked_fill(~mask, -1e4)

        if topk is not None and 0 < int(topk) < scores.size(-1):
            k = min(int(topk), scores.size(-1))
            topk_idx = scaled_scores.topk(k=k, dim=-1).indices
            topk_mask = torch.zeros_like(mask)
            topk_mask.scatter_(1, topk_idx, True)
            mask = mask & topk_mask
            scaled_scores = scaled_scores.masked_fill(~mask, -1e4)

        weights = F.softmax(scaled_scores, dim=-1)
        weights = weights * mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return weights

    def _sparsify_weights(self, weights, topk=None, mask=None):
        if topk is None or int(topk) <= 0 or int(topk) >= weights.size(-1):
            if mask is not None:
                weights = weights * mask.to(dtype=weights.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            return weights

        k = min(int(topk), weights.size(-1))
        if mask is None:
            mask = torch.ones_like(weights, dtype=torch.bool)
        else:
            mask = mask.to(device=weights.device, dtype=torch.bool)

        masked_weights = weights.masked_fill(~mask, -1.0)
        topk_idx = masked_weights.topk(k=k, dim=-1).indices
        topk_mask = torch.zeros_like(mask)
        topk_mask.scatter_(1, topk_idx, True)
        sparse = weights * (mask & topk_mask).to(dtype=weights.dtype)
        sparse = sparse / sparse.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return sparse

    def _topk_mean(self, scores, topk, mask=None):
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        k = min(int(topk), scores.size(-1))
        values = scores.topk(k=k, dim=-1).values
        if mask is not None:
            valid_counts = mask.sum(dim=-1).clamp(min=1)
            denom = valid_counts.clamp(max=k).to(dtype=scores.dtype)
        else:
            denom = torch.full(
                (scores.size(0),),
                float(k),
                device=scores.device,
                dtype=scores.dtype,
            )
        return values.sum(dim=-1) / denom

    def _normalized_entropy(self, weights, mask=None):
        if mask is not None:
            weights = weights * mask.to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            valid_count = mask.sum(dim=-1).clamp(min=2).to(dtype=weights.dtype)
            norm = torch.log(valid_count)
        else:
            norm = torch.log(
                torch.full(
                    (weights.size(0),),
                    float(max(weights.size(-1), 2)),
                    device=weights.device,
                    dtype=weights.dtype,
                )
            )
        entropy = -(weights.clamp(min=1e-8).log() * weights).sum(dim=-1)
        return entropy / norm.clamp(min=1e-6)

    def _encode_image_tokens(self, image: torch.Tensor):
        image = self._prepare_image_input(image)

        batch_size, num_frames, channels, height, width = image.shape
        flat_image = image.reshape(batch_size * num_frames, channels, height, width)

        frame_features = self.visual(flat_image)
        frame_features = frame_features.reshape(batch_size, num_frames, self.visual_dim)

        if self.frame_pool is not None and num_frames > 1:
            video_global_hidden, frame_tokens = self.frame_pool(frame_features, return_tokens=True)
        else:
            video_global_hidden = frame_features.mean(dim=1)
            frame_tokens = frame_features

        return video_global_hidden, frame_tokens

    def _compute_frame_selection_weights(self, frame_tokens, token_hidden, attention_mask, level_ids=None):
        if self.selection_pooling == "xpool":
            return self._compute_xpool_frame_selection_weights(
                frame_tokens=frame_tokens,
                token_hidden=token_hidden,
                attention_mask=attention_mask,
                level_ids=level_ids,
            )
        return self._compute_similarity_frame_selection_weights(
            frame_tokens=frame_tokens,
            token_hidden=token_hidden,
            attention_mask=attention_mask,
            level_ids=level_ids,
        )

    def _compute_similarity_frame_selection_weights(self, frame_tokens, token_hidden, attention_mask, level_ids=None):
        batch_size = frame_tokens.size(0)
        dtype = frame_tokens.dtype
        device = frame_tokens.device
        token_mask = attention_mask.bool()

        # Frame side: shared video_projection (same as main branch)
        frame_tokens = self.video_adapter(frame_tokens)
        frame_local = F.normalize(self.video_projection(frame_tokens), dim=-1)

        # Text side: mean pool → shared text.project (same as _encode_text_global)
        text_mask = token_mask.unsqueeze(-1).to(dtype=dtype)
        pooled_hidden = (token_hidden * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1e-6)
        pooled_hidden = self.text_adapter(pooled_hidden)
        text_global = F.normalize(self.text.project(pooled_hidden), dim=-1)

        # Dot product: each frame vs global text
        raw_scores = torch.einsum("btd,bd->bt", frame_local, text_global)

        frame_temps = self._get_level_values(
            level_ids,
            self.level_frame_temperatures,
            1.0,
            batch_size,
            device,
            dtype,
        )
        frame_weights = self._normalize_scores(
            raw_scores,
            temperatures=frame_temps,
        )

        top2 = raw_scores.topk(k=min(2, raw_scores.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            confidence = top2[:, 0]
        else:
            confidence = top2[:, 0] - top2[:, 1]
        confidence = (confidence / 0.25).clamp(min=0.0, max=1.0)

        return frame_weights, confidence.detach()

    def _compute_xpool_frame_selection_weights(self, frame_tokens, token_hidden, attention_mask, level_ids=None):
        batch_size = frame_tokens.size(0)
        dtype = frame_tokens.dtype
        device = frame_tokens.device

        token_mask = attention_mask.bool()
        text_mask = token_mask.unsqueeze(-1).to(dtype=token_hidden.dtype)
        pooled_hidden = (token_hidden * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1e-6)

        frame_tokens = self.video_adapter(frame_tokens)
        pooled_hidden = self.text_adapter(pooled_hidden)
        query = F.normalize(self.selection_text_query_projection(pooled_hidden), dim=-1)
        keys = F.normalize(self.selection_frame_key_projection(frame_tokens), dim=-1)
        raw_scores = torch.einsum("bd,btd->bt", query, keys)

        frame_temps = self._get_level_values(
            level_ids,
            self.level_frame_temperatures,
            1.0,
            batch_size,
            device,
            dtype,
        )
        frame_weights = self._normalize_scores(
            raw_scores,
            temperatures=frame_temps,
        )

        top2 = raw_scores.topk(k=min(2, raw_scores.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            confidence = top2[:, 0]
        else:
            confidence = top2[:, 0] - top2[:, 1]
        confidence = (confidence / 0.25).clamp(min=0.0, max=1.0)
        return frame_weights, confidence.detach()

    def _project_video_global(self, video_global_hidden):
        video_global_hidden = self.video_adapter(video_global_hidden)
        return F.normalize(self.video_projection(video_global_hidden), dim=-1)

    def _project_selected_video(self, frame_tokens, frame_weights):
        frame_tokens = self.video_adapter(frame_tokens)
        selected_hidden = torch.sum(frame_weights.unsqueeze(-1) * frame_tokens, dim=1)
        return F.normalize(self.video_projection(selected_hidden), dim=-1)

    def _encode_text_global(self, text_global_hidden):
        self.last_text_gate = None
        text_global_hidden = self.text_adapter(text_global_hidden)
        return F.normalize(self.text.project(text_global_hidden), dim=-1)

    def encode_training_pair(self, image, input_ids, attention_mask, level_ids=None, selection_image=None):
        source_image = selection_image if (self.training and selection_image is not None) else image
        if source_image is None:
            raise ValueError("encode_training_pair requires image or selection_image.")

        video_global_hidden, frame_tokens = self._encode_image_tokens(source_image)
        _, token_hidden, text_global_hidden = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True,
        )
        image_features = self._project_video_global(video_global_hidden)
        text_features = self._encode_text_global(text_global_hidden)
        selected_image_features = None

        if self.training and selection_image is not None:
            frame_weights, pair_confidence = self._compute_frame_selection_weights(
                frame_tokens=frame_tokens,
                token_hidden=token_hidden,
                attention_mask=attention_mask,
                level_ids=level_ids,
            )
            selected_image_features = self._project_selected_video(
                frame_tokens,
                frame_weights,
            )
            frame_entropy = self._normalized_entropy(frame_weights)
            self.last_frame_weights = frame_weights.detach()
            self.last_pair_confidence = pair_confidence.detach()
            self.last_frame_entropy = frame_entropy.detach()
            self.last_frame_peak = frame_weights.max(dim=-1).values.detach()
        else:
            self.last_frame_weights = None
            self.last_pair_confidence = None
            self.last_frame_entropy = None
            self.last_frame_peak = None

        self.last_token_weights = None
        self.last_pair_weights = None
        self.last_entropy_regularization = None
        self.last_distill_regularization = None
        self.last_token_entropy = None
        self.last_token_peak = None
        self.last_video_gate = None

        return image_features, selected_image_features, text_features

    def encode_image(self, image: torch.Tensor, level_ids: torch.Tensor = None):
        video_global_hidden, _ = self._encode_image_tokens(image)
        video_features = self._project_video_global(video_global_hidden)
        self.last_frame_weights = None
        self.last_frame_entropy = None
        self.last_frame_peak = None
        self.last_token_weights = None
        self.last_token_entropy = None
        self.last_token_peak = None
        self.last_text_gate = None
        self.last_pair_confidence = None
        self.last_pair_weights = None
        self.last_entropy_regularization = None
        self.last_distill_regularization = None
        self.last_video_gate = None
        return video_features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        level_ids: torch.Tensor = None,
    ):
        _, _, text_global_hidden = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True,
        )
        text_features = self._encode_text_global(text_global_hidden)
        self.last_token_weights = None
        self.last_token_entropy = None
        self.last_token_peak = None
        self.last_text_gate = None
        return F.normalize(text_features, dim=-1)

    def forward(self, image, input_ids, attention_mask, level_ids=None):
        image_features, _, text_features = self.encode_training_pair(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            level_ids=level_ids,
        )

        logit_scale = self.logit_scale.exp()
        base_logits = logit_scale * image_features @ text_features.t()
        return base_logits, base_logits.t()

    def _get_convnext_visual_parts(self):
        visual = self.visual
        base_model = visual.model if hasattr(visual, "model") else visual
        if hasattr(base_model, "get_base_model"):
            base_model = base_model.get_base_model()

        features = getattr(base_model, "features", None)
        classifier = getattr(base_model, "classifier", None)
        return features, classifier

    def _compute_trainable_stage_indices(self):
        features, _ = self._get_convnext_visual_parts()
        if features is None:
            return set()

        num_features = len(features)
        n = int(self.train_encoder_num_stages)
        if n <= 0:
            return set(range(num_features))
        return set(range(max(0, num_features - n), num_features))

    def _unfreeze_convnext_stages(self, stage_indices):
        features, classifier = self._get_convnext_visual_parts()
        if features is None:
            return False

        num_features = len(features)
        for idx in stage_indices:
            if 0 <= idx < num_features:
                for p in features[idx].parameters():
                    p.requires_grad = True

        if num_features - 1 in stage_indices and classifier is not None and len(classifier) > 0:
            for p in classifier[0].parameters():
                p.requires_grad = True

        return True

    def _set_convnext_stages_train(self, stage_indices):
        features, classifier = self._get_convnext_visual_parts()
        if features is None:
            return False

        num_features = len(features)
        for idx in stage_indices:
            if 0 <= idx < num_features:
                features[idx].train()

        if num_features - 1 in stage_indices and classifier is not None and len(classifier) > 0:
            classifier[0].train()

        return True

    def _enable_text_activation_checkpointing(self):
        text_backbone = self.text.backbone
        if hasattr(text_backbone, "gradient_checkpointing_enable"):
            text_backbone.gradient_checkpointing_enable()
        elif hasattr(getattr(text_backbone, "base_model", None), "gradient_checkpointing_enable"):
            text_backbone.base_model.gradient_checkpointing_enable()

        if hasattr(text_backbone, "config") and hasattr(text_backbone.config, "use_cache"):
            text_backbone.config.use_cache = False
        elif hasattr(getattr(text_backbone, "base_model", None), "config") and hasattr(
            text_backbone.base_model.config, "use_cache",
        ):
            text_backbone.base_model.config.use_cache = False

    def freeze_encoders_train_projections(self):
        # 1. 先全部冻结
        for p in self.visual.parameters():
            p.requires_grad = False

        for p in self.text.backbone.parameters():
            p.requires_grad = False

        stage_indices = set()

        # 2. 视觉侧：解冻最后 N 个 stage（ConvNeXt）或 block（ViT）
        if self.train_encoder_base_layers:
            if hasattr(self.visual, "unfreeze_last_n_blocks"):
                n = int(self.train_encoder_num_stages)
                if n <= 0:
                    n = len(self.visual.model.blocks)
                self.visual.unfreeze_last_n_blocks(n)
            elif hasattr(self.visual, "unfreeze_stages"):
                stage_indices = self._compute_trainable_stage_indices()
                self.visual.unfreeze_stages(stage_indices)
            elif hasattr(self.visual, "unfreeze_last_stage"):
                features, _ = self._get_convnext_visual_parts()
                if features is not None:
                    stage_indices = {len(features) - 1}
                self.visual.unfreeze_last_stage()
            else:
                stage_indices = self._compute_trainable_stage_indices()
                self._unfreeze_convnext_stages(stage_indices)

            # 3. 文本侧：放开最后两层
            for layer in self.text.backbone.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # 4. 头部继续训练
        if self.frame_pool is not None:
            for p in self.frame_pool.parameters():
                p.requires_grad = True

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

        # 5. gradient checkpointing
        if self.encoder_gradient_checkpointing:
            if hasattr(self.visual, "unfreeze_last_n_blocks"):
                n = max(1, int(self.train_encoder_num_stages))
                apply_vit_gradient_checkpointing(self.visual, n)
            elif stage_indices:
                apply_convnext_gradient_checkpointing(self.visual, stage_indices)
            self._enable_text_activation_checkpointing()

    def set_frozen_modules_eval(self):
        # 先把整块 frozen 部分设成 eval
        self.visual.eval()
        self.text.backbone.eval()

        # 再把允许微调的部分切回 train
        if self.train_encoder_base_layers:
            if hasattr(self.visual, "set_last_n_blocks_train"):
                n = max(1, int(self.train_encoder_num_stages))
                self.visual.set_last_n_blocks_train(n)
            elif hasattr(self.visual, "set_stages_train"):
                stage_indices = self._compute_trainable_stage_indices()
                self.visual.set_stages_train(stage_indices)
            elif hasattr(self.visual, "set_last_stage_train"):
                self.visual.set_last_stage_train()
            else:
                stage_indices = self._compute_trainable_stage_indices()
                self._set_convnext_stages_train(stage_indices)

            for layer in self.text.backbone.encoder.layer[-2:]:
                layer.train()

        self.video_adapter.train()
        self.text_adapter.train()

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_info(model):
    total, trainable = count_parameters(model)

    frame_pool_params = 0
    if getattr(model, "frame_pool", None) is not None:
        frame_pool_params = sum(p.numel() for p in model.frame_pool.parameters())

    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"Visual Encoder   : {getattr(model.visual, 'name', model.visual.__class__.__name__)}")
    print(f"Text Encoder     : {model.text.__class__.__name__}")
    print(f"Visual Dim       : {model.visual_dim}")
    print(f"Temporal Hidden  : {getattr(model, 'temporal_hidden_dim', model.visual_dim)}")
    print(
        f"Video Projection : "
        f"{model.video_projection.in_features} -> {model.video_projection.out_features}"
    )
    print(
        "Frame Pool       : "
        f"{'TimeSformer-style temporal pool' if model.frame_pool is not None else 'disabled'}"
    )
    print("Logit Scale      : learnable scalar")
    print("-" * 80)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print("-" * 80)
    print(f"visual params    : {sum(p.numel() for p in model.visual.parameters()):,}")
    print(f"text params      : {sum(p.numel() for p in model.text.parameters()):,}")
    print(f"video adapter params: {sum(p.numel() for p in model.video_adapter.parameters()):,}")
    print(f"text adapter params : {sum(p.numel() for p in model.text_adapter.parameters()):,}")
    print(f"video proj params: {sum(p.numel() for p in model.video_projection.parameters()):,}")
    print(f"frame pool params: {frame_pool_params:,}")
    print(f"train encoder base layers: {getattr(model, 'train_encoder_base_layers', True)}")
    print("=" * 80)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLP(
        embed_dim=256,
        text_model_name="marcobombieri/surgicberta",
        vision_backbone="convnext_lemonfm",
        vision_pretrained_weights="lemonfm.pth",
        num_frames=8,
        temporal_num_layers=2,
        temporal_num_heads=12,
        temporal_dropout=0.1,
        temporal_hidden_dim=768,
    ).to(device)

    print_model_info(model)

    save_path = "vlp_ckpt.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved initial VLP checkpoint to: {save_path}")
