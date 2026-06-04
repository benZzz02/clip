import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


PESKAVLP_ROOT = Path(__file__).resolve().parent / "third_party" / "PeskaVLP"


def _ensure_peskavlp_import_path():
    root = str(PESKAVLP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _gather_with_grad(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    rank = dist.get_rank()
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


def _gather_no_grad(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class PeskaVLPCompatibleLoss(nn.Module):
    """Feed the local VLP model into the original PeskaVLP loss family."""

    def __init__(
        self,
        temperature=0.1,
        alpha_weight=0.75,
        dtw_beta=0.0,
        dtw_ratio=0.5,
        dtw_scale_factor=0.01,
    ):
        super().__init__()
        _ensure_peskavlp_import_path()
        from codes.loss.combine_ssl_vl_new import SSL_VL_Loss_new
        from codes.loss.hierarchy_loss_2 import Abstract_Loss as HierInfoNCELoss
        from codes.loss.hierarchy_loss_2_dtw import Abstract_Loss as HierInfoNCEDTWLoss

        self.action_loss = SSL_VL_Loss_new(
            temperature=temperature,
            alpha_weight=alpha_weight,
        )
        self.mid_loss = HierInfoNCELoss(
            temperature=temperature,
            alpha_weight=alpha_weight,
        )
        self.coarse_loss = HierInfoNCEDTWLoss(
            temperature=temperature,
            alpha_weight=alpha_weight,
            dtw_beta=dtw_beta,
            dtw_ratio=dtw_ratio,
            dtw_scale_factor=dtw_scale_factor,
        )

    def _unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def _reset_debug_state(self, vlp):
        vlp.last_frame_weights = None
        vlp.last_pair_confidence = None
        vlp.last_frame_entropy = None
        vlp.last_frame_peak = None
        vlp.last_token_weights = None
        vlp.last_pair_weights = None
        vlp.last_entropy_regularization = None
        vlp.last_distill_regularization = None
        vlp.last_token_entropy = None
        vlp.last_token_peak = None
        vlp.last_video_gate = None
        vlp.last_text_gate = None

    def _encode_video(self, vlp, video):
        video_hidden, frame_tokens = vlp._encode_image_tokens(video)
        video_features = vlp._project_video_global(video_hidden)

        adapted_frame_tokens = vlp.video_adapter(frame_tokens)
        frame_features = F.normalize(
            vlp.frame_local_projection(adapted_frame_tokens),
            dim=-1,
        )
        self._reset_debug_state(vlp)
        return video_features, frame_features

    def _encode_text(self, vlp, input_ids, attention_mask):
        original_shape = input_ids.shape
        if input_ids.ndim == 3:
            batch_size, num_candidates, seq_len = original_shape
            input_ids = input_ids.reshape(batch_size * num_candidates, seq_len)
            attention_mask = attention_mask.reshape(batch_size * num_candidates, seq_len)
        else:
            batch_size = None
            num_candidates = None

        _, _, text_hidden = vlp.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True,
        )
        text_features = vlp._encode_text_global(text_hidden)

        if batch_size is not None:
            text_features = text_features.reshape(batch_size, num_candidates, -1)
        return text_features

    def _logit_scale(self, vlp):
        with torch.no_grad():
            vlp.logit_scale.clamp_(0, 4.6052)
        return vlp.logit_scale.exp()

    def forward_action(
        self,
        model,
        video,
        video_aug1,
        video_aug2,
        input_ids,
        attention_mask,
        candidate_input_ids,
        candidate_attention_mask,
    ):
        vlp = self._unwrap(model)
        video_features, _ = self._encode_video(vlp, video)
        video_aug1_features, _ = self._encode_video(vlp, video_aug1)
        video_aug2_features, _ = self._encode_video(vlp, video_aug2)
        text_features = self._encode_text(vlp, input_ids, attention_mask)
        candidate_features = self._encode_text(
            vlp,
            candidate_input_ids,
            candidate_attention_mask,
        )

        loss = self.action_loss(
            _gather_with_grad(video_features),
            _gather_with_grad(video_aug1_features),
            _gather_with_grad(video_aug2_features),
            [
                _gather_with_grad(text_features),
                _gather_with_grad(candidate_features),
            ],
            logit_scale=self._logit_scale(vlp),
        )
        return loss, {"peskavlp_compatible/action_ssl_vl": loss.detach()}

    def forward_hierarchy(
        self,
        model,
        level_name,
        video,
        input_ids,
        attention_mask,
        candidate_input_ids,
        candidate_attention_mask,
        pos_step,
    ):
        level_name = str(level_name).strip().lower()
        if level_name not in {"mid", "coarse"}:
            raise ValueError(f"forward_hierarchy expects mid or coarse, got {level_name}")

        vlp = self._unwrap(model)
        video_features, frame_features = self._encode_video(vlp, video)
        text_features = self._encode_text(vlp, input_ids, attention_mask)
        candidate_features = self._encode_text(
            vlp,
            candidate_input_ids,
            candidate_attention_mask,
        )

        loss_fn = self.mid_loss if level_name == "mid" else self.coarse_loss
        loss = loss_fn(
            _gather_with_grad(video_features),
            [
                _gather_with_grad(text_features),
                _gather_with_grad(candidate_features),
            ],
            video_embd_frame=_gather_with_grad(frame_features),
            pos_step=_gather_no_grad(pos_step.long()),
            logit_scale=self._logit_scale(vlp),
        )
        return loss, {f"peskavlp_compatible/hierarchy_{level_name}": loss.detach()}
