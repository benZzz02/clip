"""
AllGather with gradient preservation.

Extends the pattern from train.py's concat_all_gather (which uses .detach())
to support gradient flow through the gather operation — needed when multiple
loss terms (InfoNCE + NTXent + MILNCE + SimCLR) compose gradients on the
full distributed contrastive matrix.

This is a standard PyTorch autograd.Function wrapping dist.all_gather.
"""

import torch
import torch.distributed as dist


class AllGather(torch.autograd.Function):
    """all_gather with proper gradient back-propagation."""

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
        return grad_output[start:end], None, None


def all_gather(tensor, world_size, rank):
    return AllGather.apply(tensor, world_size, rank)
