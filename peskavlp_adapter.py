import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from transformers import AutoModel


def extract_state_dict(payload):
    if isinstance(payload, dict):
        if "model_state_dict" in payload:
            return payload["model_state_dict"], "model_state_dict"
        if "state_dict" in payload:
            return payload["state_dict"], "state_dict"
    return payload, "plain_state_dict"


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def load_state_dict_flexible(
    model,
    ckpt_path,
    device,
    source="checkpoint",
    validate_prefixes=False,
    allowed_missing_prefixes=(),
    allowed_unexpected_prefixes=(),
):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict, format_name = extract_state_dict(checkpoint)
    msg = model.load_state_dict(normalize_state_dict_keys(state_dict), strict=False)

    if validate_prefixes:
        disallowed_missing = [
            key for key in msg.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        disallowed_unexpected = [
            key for key in msg.unexpected_keys
            if not key.startswith(allowed_unexpected_prefixes)
        ]
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                f"{source} 与当前模型不匹配。\n"
                f"Missing keys: {msg.missing_keys}\n"
                f"Unexpected keys: {msg.unexpected_keys}"
            )

    return msg, format_name


def _build_torchvision_resnet(backbone_name: str, pretrained):
    builders = {
        "resnet_18": torchvision.models.resnet18,
        "resnet_34": torchvision.models.resnet34,
        "resnet_50": torchvision.models.resnet50,
        "resnet_101": torchvision.models.resnet101,
    }
    if backbone_name not in builders:
        raise ValueError(f"Unsupported PeskaVLP vision backbone: {backbone_name}")

    load_imagenet = str(pretrained).lower() in {"imagenet", "true", "1", "yes", "y"}
    model = builders[backbone_name](pretrained=load_imagenet)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()

    if isinstance(pretrained, str) and pretrained not in {"imagenet", "random"} and os.path.isfile(pretrained):
        payload = torch.load(pretrained, map_location="cpu")
        state_dict, _ = extract_state_dict(payload)
        model.load_state_dict(normalize_state_dict_keys(state_dict), strict=False)

    return model, feature_dim


class PeskaVLPImageEncoder(nn.Module):
    def __init__(self, embed_dim=768, pretrained="random", backbone_name="resnet_50", img_norm=False):
        super().__init__()
        self.model, feature_dim = _build_torchvision_resnet(backbone_name, pretrained)
        self.global_embedder = nn.Linear(feature_dim, embed_dim)
        self.norm = bool(img_norm)

    def forward(self, x):
        global_features = self.model(x)
        global_emb = self.global_embedder(global_features)
        if self.norm:
            global_emb = F.normalize(global_emb, dim=-1)
        return global_emb


class PeskaVLPTextEncoder(nn.Module):
    def __init__(
        self,
        text_model_name,
        tokenizer=None,
        text_last_n_layers=4,
        text_aggregate_method="sum",
        text_norm=False,
        text_embedding_dim=768,
        text_freeze_bert=False,
        text_agg_tokens=True,
    ):
        super().__init__()
        self.bert_type = text_model_name
        self.last_n_layers = int(text_last_n_layers)
        self.aggregate_method = text_aggregate_method
        self.norm = bool(text_norm)
        self.embedding_dim = int(text_embedding_dim)
        self.freeze_bert = bool(text_freeze_bert)
        self.agg_tokens = bool(text_agg_tokens)

        self.model = AutoModel.from_pretrained(
            self.bert_type,
            output_hidden_states=True,
        )

        if tokenizer is None:
            raise ValueError("PeskaVLPTextEncoder requires the tokenizer used by the training pipeline.")
        self.idxtoword = {v: k for k, v in tokenizer.get_vocab().items()}

        if self.freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        aggregated_batch = []

        for embs, caption_id in zip(embeddings, caption_ids):
            aggregated = []
            token_bank = []

            for word_emb, word_id in zip(embs, caption_id):
                word = self.idxtoword.get(int(word_id.item()), "[UNK]")

                if word == "[SEP]":
                    if token_bank:
                        aggregated.append(torch.stack(token_bank).sum(dim=0))
                        token_bank = []
                    aggregated.append(word_emb)
                    break

                if not word.startswith("##"):
                    if token_bank:
                        aggregated.append(torch.stack(token_bank).sum(dim=0))
                    token_bank = [word_emb]
                else:
                    token_bank.append(word_emb)

            if token_bank:
                aggregated.append(torch.stack(token_bank).sum(dim=0))

            if not aggregated:
                aggregated = [embs[0]]

            aggregated = torch.stack(aggregated)
            padding_size = num_words - len(aggregated)
            if padding_size > 0:
                paddings = torch.zeros(padding_size, num_layers, dim, device=aggregated.device, dtype=aggregated.dtype)
                aggregated = torch.cat([aggregated, paddings], dim=0)
            else:
                aggregated = aggregated[:num_words]

            aggregated_batch.append(aggregated)

        aggregated_batch = torch.stack(aggregated_batch)
        return aggregated_batch.permute(0, 2, 1, 3)

    def forward(self, ids=None, attn_mask=None, token_type=None):
        if token_type is None:
            token_type = torch.zeros_like(ids)

        outputs = self.model(
            input_ids=ids,
            attention_mask=attn_mask,
            token_type_ids=token_type,
            return_dict=True,
        )

        if self.last_n_layers > 1:
            hidden_states = outputs.hidden_states
            embeddings = torch.stack(hidden_states[-self.last_n_layers:]).permute(1, 0, 2, 3)
            if self.agg_tokens:
                embeddings = self.aggregate_tokens(embeddings, ids)

            sent_embeddings = embeddings.mean(dim=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(dim=1)
                sent_embeddings = sent_embeddings.sum(dim=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(dim=1)
                sent_embeddings = sent_embeddings.mean(dim=1)
            else:
                raise ValueError(f"Unsupported aggregate_method: {self.aggregate_method}")
        else:
            word_embeddings = outputs.last_hidden_state
            sent_embeddings = outputs.pooler_output
            if sent_embeddings is None:
                sent_embeddings = word_embeddings[:, 0]

        if self.norm:
            word_embeddings = F.normalize(word_embeddings, dim=-1)
            sent_embeddings = F.normalize(sent_embeddings, dim=-1)

        return word_embeddings.permute(0, 2, 1), sent_embeddings, None


class PeskaVLPAdapter(nn.Module):
    def __init__(
        self,
        text_model_name,
        tokenizer,
        embed_dim=768,
        vision_backbone_name="resnet_50",
        vision_pretrained="random",
        image_norm=False,
        text_norm=False,
        text_last_n_layers=4,
        text_aggregate_method="sum",
        text_freeze_bert=False,
        text_agg_tokens=True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.backbone_img = PeskaVLPImageEncoder(
            embed_dim=self.embed_dim,
            pretrained=vision_pretrained,
            backbone_name=vision_backbone_name,
            img_norm=image_norm,
        )
        self.backbone_text = PeskaVLPTextEncoder(
            text_model_name=text_model_name,
            tokenizer=tokenizer,
            text_last_n_layers=text_last_n_layers,
            text_aggregate_method=text_aggregate_method,
            text_norm=text_norm,
            text_embedding_dim=self.embed_dim,
            text_freeze_bert=text_freeze_bert,
            text_agg_tokens=text_agg_tokens,
        )

        self.frame_local_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.local_temperature = 0.15
        self.register_buffer(
            "level_frame_temperatures",
            torch.tensor((0.35, 0.8, 1.6), dtype=torch.float32),
            persistent=False,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        self.last_frame_weights = None
        self.last_pair_confidence = None
        self.last_frame_entropy = None
        self.last_frame_peak = None

    def configure_window_denoise(self, local_temperature: float, level_frame_temperatures):
        self.local_temperature = float(local_temperature)
        self.level_frame_temperatures = torch.tensor(
            level_frame_temperatures,
            dtype=torch.float32,
            device=self.level_frame_temperatures.device,
        )

    def enable_lora_trainable_parameters(self):
        self.backbone_img.global_embedder.weight.requires_grad = True
        if self.backbone_img.global_embedder.bias is not None:
            self.backbone_img.global_embedder.bias.requires_grad = True
        self.frame_local_projection.weight.requires_grad = True
        if self.frame_local_projection.bias is not None:
            self.frame_local_projection.bias.requires_grad = True
        self.logit_scale.requires_grad = True

    def allowed_missing_prefixes(self):
        return ("frame_local_projection.", "logit_scale")

    def allowed_unexpected_prefixes(self):
        return (
            "frame_local_projection.",
            "logit_scale",
            "backbone_text.model.embeddings.position_ids",
        )

    def _encode_frame_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W] or [B, T, C, H, W], got {tuple(images.shape)}")

        batch_size, num_frames, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * num_frames, channels, height, width)
        frame_embeddings = self.backbone_img(flat_images)
        return frame_embeddings.reshape(batch_size, num_frames, -1)

    def _encode_text_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        token_type_ids = torch.zeros_like(input_ids)
        word_embeddings, sent_embeddings, _ = self.backbone_text(
            ids=input_ids,
            attn_mask=attention_mask,
            token_type=token_type_ids,
        )
        return word_embeddings.transpose(1, 2), sent_embeddings

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        frame_embeddings = self._encode_frame_embeddings(images)
        pooled = frame_embeddings.mean(dim=1)
        return F.normalize(pooled, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, sent_embeddings = self._encode_text_hidden(input_ids, attention_mask)
        return F.normalize(sent_embeddings, dim=-1)

    def _get_level_values(self, level_ids, values, default_value, batch_size, device, dtype):
        if level_ids is None:
            return torch.full((batch_size,), float(default_value), device=device, dtype=dtype)
        level_ids = level_ids.to(device=device, dtype=torch.long).clamp(min=0, max=len(values) - 1)
        return values.to(device=device, dtype=dtype)[level_ids]

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

    def _normalize_scores(self, scores, temperatures=1.0):
        if not torch.is_tensor(temperatures):
            temperatures = torch.full(
                (scores.size(0),),
                float(temperatures),
                device=scores.device,
                dtype=scores.dtype,
            )
        temperatures = temperatures.to(device=scores.device, dtype=scores.dtype).clamp(min=1e-4)
        scaled_scores = scores / temperatures.unsqueeze(-1)
        weights = F.softmax(scaled_scores, dim=-1)
        return weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def _normalized_entropy(self, weights):
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

    def _compute_frame_selection_weights(self, frame_embeddings, text_embeds, attention_mask, level_ids=None):
        batch_size = frame_embeddings.size(0)
        dtype = frame_embeddings.dtype
        device = frame_embeddings.device
        token_mask = attention_mask.bool()

        frame_local = F.normalize(self.frame_local_projection(frame_embeddings), dim=-1)
        token_local = F.normalize(text_embeds, dim=-1)
        raw_alignment = torch.matmul(frame_local, token_local.transpose(1, 2))
        scaled_alignment = raw_alignment / max(self.local_temperature, 1e-4)

        frame_scores = self._masked_max(
            scaled_alignment,
            token_mask.unsqueeze(1),
            dim=-1,
        )
        frame_temps = self._get_level_values(
            level_ids,
            self.level_frame_temperatures,
            1.0,
            batch_size,
            device,
            dtype,
        )
        frame_weights = self._normalize_scores(frame_scores, temperatures=frame_temps)

        frame_best = self._masked_max(raw_alignment, token_mask.unsqueeze(1), dim=-1)
        frame_mean = self._masked_mean(raw_alignment, token_mask.unsqueeze(1), dim=-1)
        frame_margin = F.relu(frame_best - frame_mean)
        confidence = (frame_margin.mean(dim=-1) / 0.35).clamp(min=0.0, max=1.0)
        return frame_weights, confidence.detach()

    def encode_training_pair(self, image, input_ids, attention_mask, level_ids=None, selection_image=None):
        source_image = selection_image if (self.training and selection_image is not None) else image
        if source_image is None:
            raise ValueError("encode_training_pair requires image or selection_image.")

        frame_embeddings = self._encode_frame_embeddings(source_image)
        text_embeds, pooled_text = self._encode_text_hidden(input_ids, attention_mask)

        image_features = F.normalize(frame_embeddings.mean(dim=1), dim=-1)
        text_features = F.normalize(pooled_text, dim=-1)
        selected_image_features = None

        if self.training and selection_image is not None:
            frame_weights, pair_confidence = self._compute_frame_selection_weights(
                frame_embeddings=frame_embeddings,
                text_embeds=text_embeds,
                attention_mask=attention_mask,
                level_ids=level_ids,
            )
            selected_hidden = torch.sum(frame_weights.unsqueeze(-1) * frame_embeddings, dim=1)
            selected_image_features = F.normalize(selected_hidden, dim=-1)
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

        return image_features, selected_image_features, text_features
