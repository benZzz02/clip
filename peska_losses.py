"""
Hierarchical contrastive loss functions for video-language pre-training.

Ported and adapted from PeskaVLP (codes/loss/). Provides:

Building blocks:
  - ClipInfoCELoss:  Standard symmetric CLIP InfoNCE (cross-entropy) loss
  - NTXentLoss:      Soft cross-entropy variant (NT-Xent from SimCLR)
  - MILNCELoss:      Multi-Instance NCE for multiple text candidates per video
  - SimCLRLoss:      Video-video contrastive loss between two augmented views

Composite losses:
  - HierarchicalLossAction:  For fine/action level with triple augmentation
  - HierarchicalLossPhase:   For mid/coarse levels with cross-text alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Building Blocks
# ==============================================================================

class ClipInfoCELoss(nn.Module):
    """
    Standard symmetric CLIP InfoNCE loss.

    Computes: CE(emb_v @ emb_t^T / temperature, labels) bidirectionally.

    Ported from PeskaVLP codes/loss/clip_infonce.py
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_v, emb_t, logit_scale=None):
        """
        Args:
            emb_v: (bs, d) video embeddings
            emb_t: (bs, d) text embeddings
            logit_scale: optional learned scale factor (overrides temperature)
        Returns:
            scalar loss
        """
        bs = emb_v.shape[0]
        labels = torch.arange(bs, device=emb_v.device)

        emb_v = F.normalize(emb_v, p=2, dim=1)
        emb_t = F.normalize(emb_t, p=2, dim=1)

        if logit_scale is None:
            logits_per_image = emb_v @ emb_t.t() / self.temperature
            logits_per_text = emb_t @ emb_v.t() / self.temperature
        else:
            logits_per_image = logit_scale * emb_v @ emb_t.t()
            logits_per_text = logit_scale * emb_t @ emb_v.t()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2.0


class NTXentLoss(nn.Module):
    """
    Soft cross-entropy (NT-Xent) variant for image-text contrastive learning.

    Unlike the SimCLR NT-Xent, this only computes cross-modal pairs
    (image-text, text-image), NOT within-modal pairs (image-image, text-text).

    Ported from PeskaVLP codes/loss/ntxent.py
    """

    def __init__(self, temperature=0.1, alpha_weight=0.75):
        super().__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight

    @staticmethod
    def soft_xent(target, logits):
        """
        Soft cross-entropy loss.
        From: https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
        """
        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, norm=True, logit_scale=None):
        """
        Args:
            zis: (bs, d) first modality embeddings (e.g., video)
            zjs: (bs, d) second modality embeddings (e.g., text)
            norm: whether to L2-normalize embeddings
            logit_scale: optional learned scale factor
        Returns:
            scalar loss
        """
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        labels = F.one_hot(
            torch.arange(batch_size, dtype=torch.int64, device=hidden1.device),
            num_classes=batch_size,
        ).float()

        if logit_scale is None:
            logits_ab = hidden1 @ hidden2.t() / self.temperature
            logits_ba = hidden2 @ hidden1.t() / self.temperature
        else:
            logits_ab = logit_scale * hidden1 @ hidden2.t()
            logits_ba = logit_scale * hidden2 @ hidden1.t()

        loss_a = self.soft_xent(labels, logits_ab)
        loss_b = self.soft_xent(labels, logits_ba)

        return self.alpha_weight * loss_a + (1 - self.alpha_weight) * loss_b


class MILNCELoss(nn.Module):
    """
    Multi-Instance NCE loss.

    Handles the case where each video has MULTIPLE text candidates.
    The logit matrix is (bs, bs, n_candidates). Positive pairs use
    logsumexp over candidates; negatives include cross-batch pairs.

    Ported from PeskaVLP codes/loss/milnce.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, video_embd, text_embd, logit_scale=None):
        """
        Args:
            video_embd: (bs, d)
            text_embd: (bs * n_c, d) — flattened candidate texts
            logit_scale: optional learned scale factor
        Returns:
            scalar loss
        """
        d = video_embd.shape[-1]
        text_embd = text_embd.view(-1, d)
        x = video_embd @ text_embd.t()
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)

        # Positive: logsumexp over candidates for each diagonal pair
        eye = torch.eye(x.shape[0], device=x.device)[:, :, None]
        nominator = (x * eye).sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)

        # Denominator: logsumexp over ALL pairs (including cross-batch negatives)
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1)
        denominator = denominator.reshape(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)

        return torch.mean(denominator - nominator)


class SimCLRLoss(nn.Module):
    """
    SimCLR contrastive loss between two augmented video views.

    Computes self-supervised loss between aug1 and aug2 of the same video.
    Masks out self-similarities (diagonal) to avoid trivial solutions.

    Ported from PeskaVLP codes/loss/simclr.py
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, aug1_embed, aug2_embed, logit_scale=None):
        """
        Args:
            aug1_embed: (bs, d)
            aug2_embed: (bs, d)
            logit_scale: optional learned scale factor
        Returns:
            dict with keys: 'loss', 'ssl_loss', 'ssl_acc'
        """
        q_a = F.normalize(aug1_embed, dim=-1, p=2)
        q_b = F.normalize(aug2_embed, dim=-1, p=2)

        batch_size = q_a.size(0)
        k_a, k_b = q_a, q_b

        labels = torch.arange(batch_size, device=q_a.device)
        masks = F.one_hot(labels, batch_size) * 1e9

        if logit_scale is None:
            logits_aa = (q_a @ k_a.t()) / self.tau - masks
            logits_bb = (q_b @ k_b.t()) / self.tau - masks
            logits_ab = (q_a @ k_b.t()) / self.tau
            logits_ba = (q_b @ k_a.t()) / self.tau
        else:
            logits_aa = logit_scale * (q_a @ k_a.t()) - masks
            logits_bb = logit_scale * (q_b @ k_b.t()) - masks
            logits_ab = logit_scale * (q_a @ k_b.t())
            logits_ba = logit_scale * (q_b @ k_a.t())

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2.0

        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(labels).sum()
            acc = 100.0 * correct / batch_size

        return {"loss": loss, "ssl_loss": loss, "ssl_acc": acc}


# ==============================================================================
# Composite Hierarchical Losses
# ==============================================================================

class HierarchicalLossAction(nn.Module):
    """
    Composite loss for the action (fine-grained) level.

    Applies video-text contrastive losses to all three views (original + two
    augmentations), plus SimCLR between the two augmented views.

    Composition (matching PeskaVLP's SSL_VL_Loss_new):
        L1: InfoNCE(video_ori, primary_text) + NTXent(video_ori, primary_text)
            + MILNCE(video_ori, candidate_texts)  -> /3
        L2: Same for aug1
        L3: Same for aug2
        L4: SimCLR(aug1, aug2)
        Total = (L1 + L2 + L3 + L4) / 4
    """

    def __init__(self, temperature=0.1, alpha_weight=0.75):
        super().__init__()
        self.clip_info = ClipInfoCELoss(temperature)
        self.ntxent = NTXentLoss(temperature, alpha_weight)
        self.mil_nce = MILNCELoss()
        self.simclr = SimCLRLoss(temperature)

    def forward(self, video_embd, aug1_embd, aug2_embd,
                primary_text_embd, candidate_text_embd, logit_scale=None):
        """
        Args:
            video_embd:       (bs, d) original video embeddings
            aug1_embd:        (bs, d) first augmentation embeddings
            aug2_embd:        (bs, d) second augmentation embeddings
            primary_text_embd:(bs, d) anchor text embeddings
            candidate_text_embd: (bs * n_c, d) flattened candidate text embeddings
            logit_scale:      optional learned scale factor
        Returns:
            scalar loss
        """
        # Loss for original video
        loss1_vl = self.clip_info(video_embd, primary_text_embd, logit_scale)
        loss1_ntx = self.ntxent(video_embd, primary_text_embd, logit_scale=logit_scale)
        loss1_mil = self.mil_nce(video_embd, candidate_text_embd, logit_scale)
        loss1 = (loss1_vl + loss1_ntx + loss1_mil) / 3.0

        # Loss for augmented view 1
        loss2_vl = self.clip_info(aug1_embd, primary_text_embd, logit_scale)
        loss2_ntx = self.ntxent(aug1_embd, primary_text_embd, logit_scale=logit_scale)
        loss2_mil = self.mil_nce(aug1_embd, candidate_text_embd, logit_scale)
        loss2 = (loss2_vl + loss2_ntx + loss2_mil) / 3.0

        # Loss for augmented view 2
        loss3_vl = self.clip_info(aug2_embd, primary_text_embd, logit_scale)
        loss3_ntx = self.ntxent(aug2_embd, primary_text_embd, logit_scale=logit_scale)
        loss3_mil = self.mil_nce(aug2_embd, candidate_text_embd, logit_scale)
        loss3 = (loss3_vl + loss3_ntx + loss3_mil) / 3.0

        # SimCLR between augmented views
        ssl_result = self.simclr(aug1_embd, aug2_embd, logit_scale)
        loss4 = ssl_result["loss"]

        return (loss1 + loss2 + loss3 + loss4) / 4.0


class HierarchicalLossPhase(nn.Module):
    """
    Composite loss for the keystep/abstract (mid/coarse) levels.

    Aligns video with the summary/phase text and aggregated candidate texts.
    Also aligns the summary text with aggregated candidates (text-text).

    Composition (matching PeskaVLP's hier_infonce):
        L1: InfoNCE(video, summary_text)
        L2: NTXent(video, summary_text)
        L3: InfoNCE(video, mean(candidate_texts))
        L4: InfoNCE(summary_text, mean(candidate_texts))
        Total = (L1 + L2 + L3 + L4) / 4
    """

    def __init__(self, temperature=0.1, alpha_weight=0.5):
        super().__init__()
        self.clip_info = ClipInfoCELoss(temperature)
        self.ntxent = NTXentLoss(temperature, alpha_weight)

    def forward(self, video_embd, summary_text_embd, candidate_text_embd,
                logit_scale=None):
        """
        Args:
            video_embd:        (bs, d) video embeddings
            summary_text_embd: (bs, d) summary/phase description embeddings
            candidate_text_embd: (bs, n_c, d) candidate text embeddings
            logit_scale:       optional learned scale factor
        Returns:
            scalar loss
        """
        # InfoNCE + NTXent: video <-> summary text
        loss1 = self.clip_info(video_embd, summary_text_embd, logit_scale)
        loss2 = self.ntxent(video_embd, summary_text_embd, logit_scale=logit_scale)

        # Aggregate candidate texts via mean pooling
        aggregate_text = candidate_text_embd.mean(dim=1)

        # InfoNCE: video <-> aggregated candidates
        loss3 = self.clip_info(video_embd, aggregate_text, logit_scale)

        # InfoNCE: summary text <-> aggregated candidates (cross-text alignment)
        loss4 = self.clip_info(summary_text_embd, aggregate_text, logit_scale)

        return (loss1 + loss2 + loss3 + loss4) / 4.0
