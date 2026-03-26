
import math
import random
from collections import defaultdict

import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedMixedLevelBatchSampler(Sampler):
    """
    保证每个 rank 的每个 batch 都包含指定数量的 fine / mid / coarse 样本。

    例子：
        batch_size = 128
        level_batch_sizes = {"fine": 80, "mid": 32, "coarse": 16}
    """

    def __init__(
        self,
        dataset,
        batch_size,
        level_batch_sizes,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=42,
        drop_last=True,
    ):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.level_batch_sizes = {str(k): int(v) for k, v in level_batch_sizes.items()}
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if sum(self.level_batch_sizes.values()) != self.batch_size:
            raise ValueError(
                f"Sum(level_batch_sizes) must equal batch_size. "
                f"Got sum={sum(self.level_batch_sizes.values())}, batch_size={self.batch_size}"
            )

        self.level_order = list(self.level_batch_sizes.keys())

        self.level_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.dataset.samples):
            level = str(sample.get("level", "fine"))
            self.level_to_indices[level].append(idx)

        missing_levels = [
            level
            for level, take_n in self.level_batch_sizes.items()
            if take_n > 0 and len(self.level_to_indices.get(level, [])) == 0
        ]
        if missing_levels:
            raise ValueError(f"Dataset is missing samples for levels: {missing_levels}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rank_shard_len(self, n):
        if self.rank >= n:
            return 0
        return 1 + (n - 1 - self.rank) // self.num_replicas

    def _num_batches_for_rank(self):
        per_level_batches = []
        for level, take_n in self.level_batch_sizes.items():
            if take_n <= 0:
                continue
            rank_level_count = self._rank_shard_len(len(self.level_to_indices[level]))
            if self.drop_last:
                per_level_batches.append(rank_level_count // take_n)
            else:
                per_level_batches.append(math.ceil(rank_level_count / take_n))

        if not per_level_batches:
            return 0
        return min(per_level_batches)

    def __len__(self):
        return self._num_batches_for_rank()

    def __iter__(self):
        num_batches = self._num_batches_for_rank()
        if num_batches == 0:
            return

        level_pools = {}
        level_ptrs = {}
        level_rngs = {}

        for level_idx, level in enumerate(self.level_order):
            indices = list(self.level_to_indices[level])
            rng = random.Random(self.seed + self.epoch * 1000 + level_idx)

            if self.shuffle:
                rng.shuffle(indices)

            rank_indices = indices[self.rank::self.num_replicas]
            if len(rank_indices) == 0 and self.level_batch_sizes[level] > 0:
                raise RuntimeError(
                    f"Rank {self.rank} received zero samples for level={level}. "
                    f"Cannot build mixed-level batches."
                )

            level_pools[level] = rank_indices
            level_ptrs[level] = 0
            level_rngs[level] = rng

        def draw_from_level(level, n):
            pool = level_pools[level]
            ptr = level_ptrs[level]

            if len(pool) < n and self.drop_last:
                raise RuntimeError(
                    f"Not enough samples for level={level} on rank={self.rank}. "
                    f"Need {n}, have {len(pool)}."
                )

            out = []
            while len(out) < n:
                if ptr >= len(pool):
                    ptr = 0
                    if self.shuffle and len(pool) > 1:
                        level_rngs[level].shuffle(pool)

                take = min(n - len(out), len(pool) - ptr)
                out.extend(pool[ptr:ptr + take])
                ptr += take

            level_ptrs[level] = ptr
            return out

        batch_rng = random.Random(self.seed + self.epoch * 100000 + self.rank)

        for _ in range(num_batches):
            batch = []
            for level in self.level_order:
                take_n = self.level_batch_sizes[level]
                if take_n > 0:
                    batch.extend(draw_from_level(level, take_n))

            if self.shuffle:
                batch_rng.shuffle(batch)

            yield batch

