"""
Hierarchical loss functions matching PeskaVLP's training flow.

InfoNCE logic follows train.py's clip_contrastive_loss pattern.
NTXent, MILNCE, SimCLR are from PeskaVLP (codes/loss/).

Composite losses match PeskaVLP exactly:
  - HierarchicalLossAction = SSL_VL_Loss_new (codes/loss/combine_ssl_vl_new.py)
  - HierarchicalLossPhase  = hier_infonce (codes/loss/hierarchy_loss_2.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Building Blocks
# ==============================================================================

def infonce_loss(emb_v, emb_t, logit_scale=None, temperature=0.1):
    """
    Symmetric InfoNCE — same math as train.py's clip_contrastive_loss.
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


def ntxent_loss(zis, zjs, logit_scale=None, temperature=0.1, alpha=0.75):
    """
    NT-Xent loss — soft cross-entropy variant from PeskaVLP codes/loss/ntxent.py.
    Only computes cross-modal pairs (image-text, text-image), NOT within-modal.

    Uses one-hot targets with softXEnt instead of hard cross-entropy.
    """
    zis = F.normalize(zis, p=2, dim=1)
    zjs = F.normalize(zjs, p=2, dim=1)
    bs = zis.shape[0]

    labels = F.one_hot(torch.arange(bs, dtype=torch.int64, device=zis.device),
                       num_classes=bs).float()

    if logit_scale is None:
        logits_ab = zis @ zjs.t() / temperature
        logits_ba = zjs @ zis.t() / temperature
    else:
        logits_ab = logit_scale * zis @ zjs.t()
        logits_ba = logit_scale * zjs @ zis.t()

    # softXEnt: log_softmax then weighted sum
    def soft_xent(target, logits):
        logprobs = F.log_softmax(logits, dim=1)
        return -(target * logprobs).sum() / logits.shape[0]

    return alpha * soft_xent(labels, logits_ab) + (1 - alpha) * soft_xent(labels, logits_ba)


def milnce_loss(video_embd, text_embd):
    """
    Multi-Instance NCE: one video ↔ N text candidates.
    From PeskaVLP codes/loss/milnce.py
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
    SimCLR contrastive loss between two augmented views.
    From PeskaVLP codes/loss/simclr.py
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
# Composite Losses — match PeskaVLP exactly
# ==============================================================================

class HierarchicalLossAction(nn.Module):
    """
    Action (fine) level — matches PeskaVLP's SSL_VL_Loss_new.

    For each of 3 video views: InfoNCE + NTXent + MILNCE → /3
    Plus: SimCLR(aug1, aug2)
    Total = mean(4 terms)
    """

    def __init__(self, temperature=0.1, ntxent_alpha=0.75):
        super().__init__()
        self.tau = temperature
        self.alpha = ntxent_alpha

    def forward(self, video_emb, aug1_emb, aug2_emb,
                primary_text_emb, candidate_text_emb, logit_scale=None):
        def _per_view(v, t):
            return (infonce_loss(v, t, logit_scale, self.tau) +
                    ntxent_loss(v, t, logit_scale, self.tau, self.alpha) +
                    milnce_loss(v, candidate_text_emb)) / 3.0

        l_ori = _per_view(video_emb, primary_text_emb)
        l_a1 = _per_view(aug1_emb, primary_text_emb)
        l_a2 = _per_view(aug2_emb, primary_text_emb)
        l_ssl = simclr_loss(aug1_emb, aug2_emb, logit_scale, self.tau)

        return (l_ori + l_a1 + l_a2 + l_ssl) / 4.0


class HierarchicalLossPhase(nn.Module):
    """
    Keystep/Abstract (mid/coarse) level — matches PeskaVLP's hier_infonce.

    4 terms:
      InfoNCE(video, summary) + NTXent(video, summary)
      + InfoNCE(video, agg_candidates) + InfoNCE(summary, agg_candidates)
    """

    def __init__(self, temperature=0.1, ntxent_alpha=0.5):
        super().__init__()
        self.tau = temperature
        self.alpha = ntxent_alpha

    def forward(self, video_emb, summary_text_emb, candidate_text_emb,
                logit_scale=None):
        agg = candidate_text_emb.mean(dim=1)

        return (infonce_loss(video_emb, summary_text_emb, logit_scale, self.tau) +
                ntxent_loss(video_emb, summary_text_emb, logit_scale, self.tau, self.alpha) +
                infonce_loss(video_emb, agg, logit_scale, self.tau) +
                infonce_loss(summary_text_emb, agg, logit_scale, self.tau)) / 4.0
