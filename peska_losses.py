"""
Composite loss functions for hierarchical VLP training.

InfoNCE logic (cross-entropy on video↔text similarity matrix) is inlined
directly — same as train.py's clip_contrastive_loss. Only MILNCE and SimCLR
add net-new functionality not present in the original CLIP repo.

Composite losses for PeskaVLP's hierarchical training levels:
  - HierarchicalLossAction: InfoNCE×3 + MILNCE×3 + SimCLR (action/fine)
  - HierarchicalLossPhase:  InfoNCE×3 (video↔text, video↔cand, text↔cand)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def infonce_loss(emb_v, emb_t, logit_scale=None, temperature=0.1):
    """
    Symmetric InfoNCE — same math as train.py's clip_contrastive_loss.
    Inlined as a function (not a class) to keep it light.

    train.py reference:
        logits = logit_scale * image_features @ all_text_features.t()
        loss = (CE(logits, labels) + CE(logits^T, labels)) / 2
    """
    bs = emb_v.shape[0]
    labels = torch.arange(bs, device=emb_v.device)

    emb_v = F.normalize(emb_v, p=2, dim=1)
    emb_t = F.normalize(emb_t, p=2, dim=1)

    if logit_scale is None:
        logits_vt = emb_v @ emb_t.t() / temperature
        logits_tv = emb_t @ emb_v.t() / temperature
    else:
        logits_vt = logit_scale * emb_v @ emb_t.t()
        logits_tv = logit_scale * emb_t @ emb_v.t()

    return (F.cross_entropy(logits_vt, labels) +
            F.cross_entropy(logits_tv, labels)) / 2.0


def milnce_loss(video_embd, text_embd):
    """
    Multi-Instance NCE: one video ↔ N text candidates.
    Positives = logsumexp over candidate dim; negatives = all cross-batch pairs.

    This is the only net-new loss not in the CLIP repo.
    """
    d = video_embd.shape[-1]
    text_embd = text_embd.reshape(-1, d)
    x = video_embd @ text_embd.t()
    x = x.view(video_embd.shape[0], video_embd.shape[0], -1)

    eye = torch.eye(x.shape[0], device=x.device).unsqueeze(-1)
    nominator = torch.logsumexp((x * eye).sum(dim=1), dim=1)
    denominator = torch.cat([x, x.permute(1, 0, 2)], dim=1)
    denominator = torch.logsumexp(denominator.reshape(x.shape[0], -1), dim=1)

    return torch.mean(denominator - nominator)


def simclr_loss(aug1, aug2, logit_scale=None, temperature=0.1):
    """
    SimCLR between two augmented views. Masks self-pairs.
    train.py doesn't have this (it doesn't do video augmentation).
    """
    q_a = F.normalize(aug1, dim=-1, p=2)
    q_b = F.normalize(aug2, dim=-1, p=2)
    bs = q_a.size(0)
    labels = torch.arange(bs, device=q_a.device)
    mask = F.one_hot(labels, bs) * 1e9

    if logit_scale is None:
        logits_aa = (q_a @ q_a.t()) / temperature - mask
        logits_bb = (q_b @ q_b.t()) / temperature - mask
        logits_ab = (q_a @ q_b.t()) / temperature
        logits_ba = (q_b @ q_a.t()) / temperature
    else:
        logits_aa = logit_scale * (q_a @ q_a.t()) - mask
        logits_bb = logit_scale * (q_b @ q_b.t()) - mask
        logits_ab = logit_scale * (q_a @ q_b.t())
        logits_ba = logit_scale * (q_b @ q_a.t())

    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
    return (loss_a + loss_b) / 2.0


# ==============================================================================
# Composite Losses (thin composition, no new primitives)
# ==============================================================================

class HierarchicalLossAction(nn.Module):
    """
    Action (fine) level: triple augmentation + multi-text MILNCE.

    For each of 3 video views: infonce(v, text) + milnce(v, candidates)
    Plus: simclr(aug1, aug2)
    → mean of 4 terms (matching PeskaVLP's SSL_VL_Loss_new).
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, video_emb, aug1_emb, aug2_emb,
                primary_text_emb, candidate_text_emb, logit_scale=None):
        def _vt_loss(v, t):
            return (infonce_loss(v, t, logit_scale, self.tau) +
                    milnce_loss(v, candidate_text_emb)) / 2.0

        l_ori = _vt_loss(video_emb, primary_text_emb)
        l_a1 = _vt_loss(aug1_emb, primary_text_emb)
        l_a2 = _vt_loss(aug2_emb, primary_text_emb)
        l_ssl = simclr_loss(aug1_emb, aug2_emb, logit_scale, self.tau)

        return (l_ori + l_a1 + l_a2 + l_ssl) / 4.0


class HierarchicalLossPhase(nn.Module):
    """
    Keystep/Abstract (mid/coarse) level: no augmentation.

    infonce(video, summary) + infonce(video, agg_candidates)
    + infonce(summary, agg_candidates) → mean of 3 terms.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, video_emb, summary_text_emb, candidate_text_emb,
                logit_scale=None):
        agg = candidate_text_emb.mean(dim=1)
        return (infonce_loss(video_emb, summary_text_emb, logit_scale, self.tau) +
                infonce_loss(video_emb, agg, logit_scale, self.tau) +
                infonce_loss(summary_text_emb, agg, logit_scale, self.tau)) / 3.0
