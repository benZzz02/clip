"""
Hierarchical loss functions for video-language pre-training.

Built on the same contrastive patterns as train.py's clip_contrastive_loss:
  - InfoNCE:  video @ text^T → symmetric cross-entropy (same as clip_contrastive_loss)
  - MILNCE:   video @ candidates^T → logsumexp over candidates (multi-text extension)
  - SimCLR:   aug1 @ aug2^T → self-supervised contrastive (augmentation pairing)

Composite losses for PeskaVLP's hierarchical training:
  - HierarchicalLossAction: InfoNCE + MILNCE + SimCLR (action/fine level)
  - HierarchicalLossPhase:  InfoNCE(v,t) + InfoNCE(v, agg_cand) + InfoNCE(t, agg_cand)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Building Blocks (following train.py's clip_contrastive_loss logic)
# ==============================================================================

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss — same logic as train.py's clip_contrastive_loss.

    Computes cross-entropy on the full (video, text) similarity matrix
    in both directions: video→text and text→video.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_v, emb_t, logit_scale=None):
        """
        Args:
            emb_v: (bs, d) L2-normalized video embeddings
            emb_t: (bs, d) L2-normalized text embeddings
            logit_scale: optional learned scale (e.g., model.logit_scale.exp())
        """
        bs = emb_v.shape[0]
        labels = torch.arange(bs, device=emb_v.device)

        emb_v = F.normalize(emb_v, p=2, dim=1)
        emb_t = F.normalize(emb_t, p=2, dim=1)

        if logit_scale is None:
            logits_vt = emb_v @ emb_t.t() / self.temperature
            logits_tv = emb_t @ emb_v.t() / self.temperature
        else:
            logits_vt = logit_scale * emb_v @ emb_t.t()
            logits_tv = logit_scale * emb_t @ emb_v.t()

        return (F.cross_entropy(logits_vt, labels) +
                F.cross_entropy(logits_tv, labels)) / 2.0


class MILNCELoss(nn.Module):
    """
    Multi-Instance NCE loss for multiple text candidates per video.

    Extends the InfoNCE pattern: instead of one text per video,
    there are N candidates. Positives use logsumexp over candidates;
    negatives include all cross-batch pairs.
    """

    def forward(self, video_embd, text_embd, logit_scale=None):
        """
        Args:
            video_embd: (bs, d)
            text_embd: (bs * n_candidates, d) — flattened
        Returns:
            scalar loss
        """
        d = video_embd.shape[-1]
        text_embd = text_embd.reshape(-1, d)
        x = video_embd @ text_embd.t()
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)

        # Positives: logsumexp over candidate dim for diagonal pairs
        eye = torch.eye(x.shape[0], device=x.device).unsqueeze(-1)
        nominator = torch.logsumexp((x * eye).sum(dim=1), dim=1)

        # Denominator: all pairs (including cross-batch)
        denominator = torch.cat([x, x.permute(1, 0, 2)], dim=1)
        denominator = torch.logsumexp(denominator.reshape(x.shape[0], -1), dim=1)

        return torch.mean(denominator - nominator)


class SimCLRLoss(nn.Module):
    """
    Self-supervised contrastive loss between two augmented video views.

    Masks out self-pairs (diagonal) to prevent trivial solutions.
    Loss = (CE([ab, aa], labels) + CE([ba, bb], labels)) / 2
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, aug1, aug2, logit_scale=None):
        q_a = F.normalize(aug1, dim=-1, p=2)
        q_b = F.normalize(aug2, dim=-1, p=2)
        bs = q_a.size(0)

        labels = torch.arange(bs, device=q_a.device)
        mask = F.one_hot(labels, bs) * 1e9

        if logit_scale is None:
            logits_aa = (q_a @ q_a.t()) / self.tau - mask
            logits_bb = (q_b @ q_b.t()) / self.tau - mask
            logits_ab = (q_a @ q_b.t()) / self.tau
            logits_ba = (q_b @ q_a.t()) / self.tau
        else:
            logits_aa = logit_scale * (q_a @ q_a.t()) - mask
            logits_bb = logit_scale * (q_b @ q_b.t()) - mask
            logits_ab = logit_scale * (q_a @ q_b.t())
            logits_ba = logit_scale * (q_b @ q_a.t())

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        return (loss_a + loss_b) / 2.0


# ==============================================================================
# Composite Hierarchical Losses
# ==============================================================================

class HierarchicalLossAction(nn.Module):
    """
    Action (fine) level loss with triple augmentation.

    For each of the 3 video views (original + aug1 + aug2):
      - InfoNCE with primary text
      - MILNCE with candidate texts (from adjacent levels)
    Plus SimCLR between aug1 and aug2.

    Total = mean(4 terms), matching PeskaVLP's SSL_VL_Loss_new.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.infonce = InfoNCELoss(temperature)
        self.milnce = MILNCELoss()
        self.simclr = SimCLRLoss(temperature)

    def forward(self, video_emb, aug1_emb, aug2_emb,
                primary_text_emb, candidate_text_emb, logit_scale=None):
        # Term 1: original video
        loss_ori = (self.infonce(video_emb, primary_text_emb, logit_scale) +
                    self.milnce(video_emb, candidate_text_emb, logit_scale)) / 2.0

        # Term 2: augmentation view 1
        loss_aug1 = (self.infonce(aug1_emb, primary_text_emb, logit_scale) +
                     self.milnce(aug1_emb, candidate_text_emb, logit_scale)) / 2.0

        # Term 3: augmentation view 2
        loss_aug2 = (self.infonce(aug2_emb, primary_text_emb, logit_scale) +
                     self.milnce(aug2_emb, candidate_text_emb, logit_scale)) / 2.0

        # Term 4: SimCLR between augmented views
        loss_ssl = self.simclr(aug1_emb, aug2_emb, logit_scale)

        return (loss_ori + loss_aug1 + loss_aug2 + loss_ssl) / 4.0


class HierarchicalLossPhase(nn.Module):
    """
    Keystep/Abstract (mid/coarse) level loss.

    - InfoNCE: video ↔ summary text
    - InfoNCE: video ↔ aggregated candidates
    - InfoNCE: summary text ↔ aggregated candidates (cross-text alignment)

    Total = mean(3 terms), matching PeskaVLP's hier_infonce.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.infonce = InfoNCELoss(temperature)

    def forward(self, video_emb, summary_text_emb, candidate_text_emb,
                logit_scale=None):
        # Video ↔ summary text
        loss_vt = self.infonce(video_emb, summary_text_emb, logit_scale)

        # Aggregate candidates via mean pooling
        agg_candidates = candidate_text_emb.mean(dim=1)  # (bs, d)

        # Video ↔ aggregated candidates
        loss_vc = self.infonce(video_emb, agg_candidates, logit_scale)

        # Summary text ↔ aggregated candidates
        loss_tc = self.infonce(summary_text_emb, agg_candidates, logit_scale)

        return (loss_vt + loss_vc + loss_tc) / 3.0
