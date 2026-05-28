"""
Gradient-preserving AllGather for distributed contrastive learning.

Ported from PeskaVLP's utils.py. Unlike the CLIP repo's concat_all_gather
(which detaches gradients and re-attaches locally), this autograd Function
preserves full gradient flow through the gather operation — critical for
multi-term loss composition (InfoNCE + NTXent + MILNCE + SimCLR) where
gradients must flow correctly from the global contrastive matrix back to
each GPU's local embeddings.
"""

import torch
import torch.distributed as dist


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor.

    Forward: gathers tensors from all GPUs and concatenates along dim 0.
    Backward: scatters gradients back to the correct GPU's local slice.
    """

    @staticmethod
    def forward(ctx, tensor, world_size, rank):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.batch_size * ctx.rank
        end = start + ctx.batch_size
        return (
            grad_output[start:end],
            None,  # world_size
            None,  # rank
        )


def all_gather(tensor, world_size, rank):
    """Convenience wrapper for AllGather autograd function."""
    return AllGather.apply(tensor, world_size, rank)
