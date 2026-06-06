import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.weight = linear.weight
        self.bias = linear.bias
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, self.out_features, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def lora_parameters(self):
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        update = self.lora_up(self.lora_down(self.lora_dropout(x))) * self.scaling
        return base + update


class LoRAEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.weight = embedding.weight
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Embedding(
            self.num_embeddings,
            self.rank,
            padding_idx=self.padding_idx,
            max_norm=None,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=False,
        )
        self.lora_up = nn.Linear(self.rank, self.embedding_dim, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.normal_(self.lora_down.weight, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.lora_down.weight[self.padding_idx].zero_()
        nn.init.zeros_(self.lora_up.weight)

    def lora_parameters(self):
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()

    def forward(self, x):
        base = F.embedding(
            x,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        update = self.lora_up(self.lora_dropout(self.lora_down(x))) * self.scaling
        return base + update


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        self._reversed_padding_repeated_twice = conv._reversed_padding_repeated_twice
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.weight = conv.weight
        self.bias = conv.bias
        self.lora_dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(
            self.in_channels,
            self.rank,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=False,
            padding_mode=self.padding_mode,
        )
        self.lora_up = nn.Conv2d(self.rank, self.out_channels, kernel_size=1, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def lora_parameters(self):
        yield from self.lora_down.parameters()
        yield from self.lora_up.parameters()

    def forward(self, x):
        if self.padding_mode != "zeros":
            base_input = F.pad(
                x,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            base_padding = (0, 0)
        else:
            base_input = x
            base_padding = self.padding
        base = F.conv2d(
            base_input,
            self.weight,
            self.bias,
            self.stride,
            base_padding,
            self.dilation,
            self.groups,
        )
        update = self.lora_up(self.lora_down(self.lora_dropout(x))) * self.scaling
        return base + update


LORA_MODULE_TYPES = (LoRALinear, LoRAEmbedding, LoRAConv2d)


def inject_lora_modules(module, rank: int, alpha: float = None, dropout: float = 0.0):
    rank = int(rank)
    if rank <= 0:
        return {"modules": 0, "params": 0}
    alpha = float(alpha if alpha and alpha > 0 else 2 * rank)
    dropout = float(dropout)

    injected_modules = 0
    injected_params = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LORA_MODULE_TYPES):
            continue
        if isinstance(child, nn.Linear):
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, name, wrapped)
            injected_modules += 1
            injected_params += sum(p.numel() for p in wrapped.lora_parameters())
        elif isinstance(child, nn.Embedding):
            wrapped = LoRAEmbedding(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, name, wrapped)
            injected_modules += 1
            injected_params += sum(p.numel() for p in wrapped.lora_parameters())
        elif isinstance(child, nn.Conv2d):
            wrapped = LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, name, wrapped)
            injected_modules += 1
            injected_params += sum(p.numel() for p in wrapped.lora_parameters())
        else:
            child_summary = inject_lora_modules(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            injected_modules += child_summary["modules"]
            injected_params += child_summary["params"]

    return {"modules": injected_modules, "params": injected_params}


def iter_lora_modules(module):
    for child in module.modules():
        if isinstance(child, LORA_MODULE_TYPES):
            yield child


def mark_lora_trainable(module, trainable: bool = True):
    for child in iter_lora_modules(module):
        for param in child.lora_parameters():
            param.requires_grad = bool(trainable)


def set_lora_modules_train(module, mode: bool = True):
    for child in iter_lora_modules(module):
        child.train(mode)


def count_lora_parameters(module):
    return sum(
        param.numel()
        for child in iter_lora_modules(module)
        for param in child.lora_parameters()
    )
