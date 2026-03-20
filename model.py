import os
import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel


def build_LemonFM(pretrained_weights="lemonfm.pth"):
    net = torchvision.models.convnext_large()
    in_dim = net.classifier[2].in_features
    net.classifier[2] = nn.Identity()

    if pretrained_weights is None:
        raise ValueError("pretrained_weights is None")

    if not os.path.isfile(pretrained_weights):
        raise FileNotFoundError(f"Local checkpoint not found: {pretrained_weights}")

    print(f"Loading LemonFM weights from local file: {os.path.abspath(pretrained_weights)}")

    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = state_dict["teacher"]
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    msg = net.load_state_dict(state_dict, strict=False)
    print(msg)

    first_key = next(iter(state_dict))
    assert torch.equal(
        net.state_dict()[first_key].cpu(),
        state_dict[first_key].cpu()
    ), f"Local checkpoint not actually loaded for key: {first_key}"

    print(f"Verified local checkpoint loaded into model for key: {first_key}")

    net.output_dim = in_dim
    return net


class SurgicBERTaTextEncoder(nn.Module):
    def __init__(self, model_name="marcobombieri/surgicberta", embed_dim=512):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
        )

        hidden_size = self.backbone.config.hidden_size
        self.text_projection = nn.Linear(hidden_size, embed_dim, bias=False)

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        text_features = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask,
        )
        text_features = self.text_projection(text_features)
        return text_features


class FrameAttentionPool(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, frame_features: torch.Tensor):
        if frame_features.ndim != 3:
            raise ValueError(
                f"Expected frame_features shape [B, T, D], got {tuple(frame_features.shape)}"
            )

        scores = self.score(frame_features).squeeze(-1)   # [B, T]
        weights = torch.softmax(scores, dim=1)            # [B, T]

        weighted = (weights.unsqueeze(-1) * frame_features).sum(dim=1)  # [B, D]
        mean_feat = frame_features.mean(dim=1)                           # [B, D]

        # 用 mean pooling 做残差，避免 gate 一开始就学成极端单帧选择
        fused = 0.5 * weighted + 0.5 * mean_feat
        fused = self.norm(fused)
        return fused


class VLP(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        text_model_name="marcobombieri/surgicberta",
        vision_pretrained_weights="lemonfm.pth",
        num_frames=4,
        temporal_num_layers=2,
        temporal_num_heads=8,
        temporal_dropout=0.1,
    ):
        super().__init__()

        self.visual = build_LemonFM(vision_pretrained_weights)
        self.text = SurgicBERTaTextEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim,
        )

        self.visual_dim = self.visual.output_dim
        self.embed_dim = embed_dim
        self.num_frames = max(1, int(num_frames))

        self.frame_pool = None
        if self.num_frames > 1:
            self.frame_pool = FrameAttentionPool(self.visual_dim)

        self.video_projection = nn.Linear(
            self.visual_dim,
            embed_dim,
            bias=False,
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        nn.init.normal_(self.video_projection.weight, std=self.visual_dim ** -0.5)

    def _prepare_image_input(self, image: torch.Tensor):
        if image.ndim == 4:
            image = image.unsqueeze(1)
        elif image.ndim != 5:
            raise ValueError(
                f"Expected image shape [B, C, H, W] or [B, T, C, H, W], got {tuple(image.shape)}"
            )
        return image

    def encode_image(self, image: torch.Tensor):
        image = self._prepare_image_input(image)

        batch_size, num_frames, channels, height, width = image.shape
        flat_image = image.reshape(batch_size * num_frames, channels, height, width)

        frame_features = self.visual(flat_image)
        frame_features = frame_features.reshape(batch_size, num_frames, self.visual_dim)

        if self.frame_pool is not None:
            video_hidden = self.frame_pool(frame_features)
        else:
            video_hidden = frame_features.mean(dim=1)

        video_features = self.video_projection(video_hidden)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        return video_features

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def forward(self, image, input_ids, attention_mask):
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def freeze_encoders_train_projections(self):
        for p in self.visual.parameters():
            p.requires_grad = False

        for p in self.text.backbone.parameters():
            p.requires_grad = False

        if self.frame_pool is not None:
            for p in self.frame_pool.parameters():
                p.requires_grad = True

        for p in self.video_projection.parameters():
            p.requires_grad = True

        for p in self.text.text_projection.parameters():
            p.requires_grad = True

        self.logit_scale.requires_grad = True

    def set_frozen_modules_eval(self):
        self.visual.eval()
        self.text.backbone.eval()


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
    print("Visual Encoder   : LemonFM (ConvNeXt-Large)")
    print(f"Text Encoder     : {model.text.__class__.__name__}")
    print(f"Visual Dim       : {model.visual_dim}")
    print(
        f"Video Projection : "
        f"{model.video_projection.in_features} -> {model.video_projection.out_features}"
    )
    print(f"Frame Pool       : {'enabled' if model.frame_pool is not None else 'disabled'}")
    print("Logit Scale      : learnable scalar")
    print("-" * 80)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print("-" * 80)
    print(f"visual params    : {sum(p.numel() for p in model.visual.parameters()):,}")
    print(f"text params      : {sum(p.numel() for p in model.text.parameters()):,}")
    print(f"video proj params: {sum(p.numel() for p in model.video_projection.parameters()):,}")
    print(f"frame pool params: {frame_pool_params:,}")
    print("=" * 80)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLP(
        embed_dim=512,
        text_model_name="marcobombieri/surgicberta",
        vision_pretrained_weights="lemonfm.pth",
        num_frames=4,
        temporal_num_layers=2,
        temporal_num_heads=8,
        temporal_dropout=0.1,
    ).to(device)

    print_model_info(model)

    save_path = "vlp_ckpt.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved initial VLP checkpoint to: {save_path}")
