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


class PeskaVLPPretrainingLoss(nn.Module):
    """Adapter that feeds local VLP embeddings into PeskaVLP pretraining losses."""

    def __init__(
        self,
        temperature=0.1,
        alpha_weight=0.75,
        dtw_beta=0.0,
        dtw_ratio=0.5,
        dtw_scale_factor=0.01,
        max_candidates=8,
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
        self.hierarchy_loss = HierInfoNCELoss(
            temperature=temperature,
            alpha_weight=alpha_weight,
        )
        self.hierarchy_dtw_loss = HierInfoNCEDTWLoss(
            temperature=temperature,
            alpha_weight=alpha_weight,
            dtw_beta=dtw_beta,
            dtw_ratio=dtw_ratio,
            dtw_scale_factor=dtw_scale_factor,
        )
        self.max_candidates = max(1, int(max_candidates))

    def _encode_local_vlp(
        self,
        vlp,
        images,
        input_ids,
        attention_mask,
        level_ids,
    ):
        video_hidden, frame_tokens = vlp._encode_image_tokens(images)
        text_features, token_hidden, text_hidden = vlp.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True,
        )

        video_features = vlp._project_video_global(video_hidden)
        text_features = vlp._encode_text_global(text_hidden)

        adapted_frame_tokens = vlp.video_adapter(frame_tokens)
        frame_features = F.normalize(vlp.frame_local_projection(adapted_frame_tokens), dim=-1)

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

        return {
            "video": video_features,
            "frame": frame_features,
            "text": text_features,
        }

    def _candidate_positions(self, level_ids, sample_indices, dataset_samples):
        batch_size = int(level_ids.numel())
        if sample_indices is None:
            return [[i] for i in range(batch_size)]

        by_video = {}
        samples = []
        for pos, sample_idx in enumerate(sample_indices.detach().cpu().tolist()):
            sample = dataset_samples[int(sample_idx)] if int(sample_idx) >= 0 else {}
            samples.append(sample)
            video_path = sample.get("video_path")
            if video_path:
                by_video.setdefault(video_path, []).append(pos)

        level_names = {0: "fine", 1: "mid", 2: "coarse"}
        out = []
        for pos, sample in enumerate(samples):
            video_path = sample.get("video_path")
            same_video = by_video.get(video_path, [pos])
            level_name = level_names.get(int(level_ids[pos].item()), "fine")

            if level_name == "coarse":
                preferred = {"mid", "fine"}
            elif level_name == "mid":
                preferred = {"fine"}
            else:
                preferred = {"fine"}

            positions = [
                other_pos
                for other_pos in same_video
                if level_names.get(int(level_ids[other_pos].item()), "fine") in preferred
            ]
            if not positions:
                positions = [pos]
            out.append(positions[: self.max_candidates])

        return out

    def _candidate_text(self, text_features, level_ids, sample_indices, dataset_samples, anchor_positions):
        all_candidate_positions = self._candidate_positions(level_ids, sample_indices, dataset_samples)
        candidate_features = []
        pos_indices = []

        for anchor_pos in anchor_positions.detach().cpu().tolist():
            positions = all_candidate_positions[int(anchor_pos)]
            features = [text_features[pos] for pos in positions]
            valid = list(range(len(features)))

            while len(features) < self.max_candidates:
                features.append(features[-1])

            candidate_features.append(torch.stack(features[: self.max_candidates], dim=0))
            pos_indices.append(
                torch.tensor(
                    valid[: self.max_candidates]
                    + [-1] * (self.max_candidates - len(valid[: self.max_candidates])),
                    device=text_features.device,
                    dtype=torch.long,
                )
            )

        return torch.stack(candidate_features, dim=0), torch.stack(pos_indices, dim=0)

    def _loss_inputs(self, encoded, level_ids, sample_indices, dataset_samples, mask):
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        video = encoded["video"][positions]
        frame = encoded["frame"][positions]
        text = encoded["text"][positions]
        candidates, pos_indices = self._candidate_text(
            encoded["text"],
            level_ids,
            sample_indices,
            dataset_samples,
            positions,
        )
        return {
            "video": _gather_with_grad(video),
            "frame": _gather_with_grad(frame),
            "text": _gather_with_grad(text),
            "candidates": _gather_with_grad(candidates),
            "pos_indices": _gather_no_grad(pos_indices),
        }

    def forward(
        self,
        model,
        images,
        input_ids,
        attention_mask,
        level_ids,
        sample_indices,
        dataset_samples,
        active_level_ids=None,
    ):
        vlp = model.module if hasattr(model, "module") else model
        encoded = self._encode_local_vlp(
            vlp=vlp,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            level_ids=level_ids,
        )

        active_mask = torch.ones_like(level_ids, dtype=torch.bool)
        if active_level_ids is not None:
            active_ids = torch.tensor(tuple(active_level_ids), device=level_ids.device, dtype=level_ids.dtype)
            active_mask = (level_ids.unsqueeze(-1) == active_ids.unsqueeze(0)).any(dim=-1)

        losses = []
        loss_parts = {}
        logit_scale = vlp.logit_scale.exp()

        fine_mask = active_mask & (level_ids == 0)
        if fine_mask.any():
            inputs = self._loss_inputs(encoded, level_ids, sample_indices, dataset_samples, fine_mask)
            loss = self.action_loss(
                inputs["video"],
                inputs["video"],
                inputs["video"],
                [inputs["text"], inputs["candidates"]],
                logit_scale=logit_scale,
            )
            losses.append(loss)
            loss_parts["peskavlp/action_ssl_vl"] = loss.detach()

        mid_mask = active_mask & (level_ids == 1)
        if mid_mask.any():
            inputs = self._loss_inputs(encoded, level_ids, sample_indices, dataset_samples, mid_mask)
            loss = self.hierarchy_loss(
                inputs["video"],
                [inputs["text"], inputs["candidates"]],
                video_embd_frame=inputs["frame"],
                pos_step=inputs["pos_indices"],
                logit_scale=logit_scale,
            )
            losses.append(loss)
            loss_parts["peskavlp/hierarchy_mid"] = loss.detach()

        coarse_mask = active_mask & (level_ids == 2)
        if coarse_mask.any():
            inputs = self._loss_inputs(encoded, level_ids, sample_indices, dataset_samples, coarse_mask)
            loss = self.hierarchy_dtw_loss(
                inputs["video"],
                [inputs["text"], inputs["candidates"]],
                video_embd_frame=inputs["frame"],
                pos_step=inputs["pos_indices"],
                logit_scale=logit_scale,
            )
            losses.append(loss)
            loss_parts["peskavlp/hierarchy_coarse_dtw"] = loss.detach()

        if not losses:
            total = encoded["video"].sum() * 0.0
        else:
            total = torch.stack(losses).sum()
        loss_parts["peskavlp/total"] = total.detach()
        return total, loss_parts
