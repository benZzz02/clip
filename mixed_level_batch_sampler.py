
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
        require_complete_triplet=False,
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
        self.require_complete_triplet = bool(require_complete_triplet)
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

        self.complete_triplets = self._build_complete_triplets()
        if self.require_complete_triplet:
            required_levels = ("fine", "mid", "coarse")
            for level in required_levels:
                if self.level_batch_sizes.get(level, 0) <= 0:
                    raise ValueError(
                        f"require_complete_triplet=True requires level_batch_sizes[{level!r}] > 0"
                    )
            if not self.complete_triplets:
                raise ValueError("No complete fine-mid-coarse triplets found in dataset.")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rank_shard_len(self, n):
        if self.rank >= n:
            return 0
        return 1 + (n - 1 - self.rank) // self.num_replicas

    def _min_rank_shard_len(self, n):
        """Return the smallest number of samples any rank receives for a level."""
        if self.num_replicas <= 0:
            return 0
        min_rank = self.num_replicas - 1
        if min_rank >= n:
            return 0
        return 1 + (n - 1 - min_rank) // self.num_replicas

    def _num_batches_for_rank(self):
        per_level_batches = []
        for level, take_n in self.level_batch_sizes.items():
            if take_n <= 0:
                continue
            # Use the minimum per-level shard count so every rank can run the same number of batches.
            rank_level_count = self._min_rank_shard_len(len(self.level_to_indices[level]))
            if self.drop_last:
                per_level_batches.append(rank_level_count // take_n)
            else:
                per_level_batches.append(math.ceil(rank_level_count / take_n))

        if not per_level_batches:
            return 0

        return min(per_level_batches)

    def _build_complete_triplets(self):
        samples_by_video = defaultdict(lambda: {"fine": [], "mid": [], "coarse": []})

        for idx, sample in enumerate(self.dataset.samples):
            level = str(sample.get("level", "fine"))
            if level not in ("fine", "mid", "coarse"):
                continue
            video_path = sample.get("video_path")
            if not video_path:
                continue
            samples_by_video[video_path][level].append((idx, sample))

        triplets = []
        for level_groups in samples_by_video.values():
            fines = sorted(level_groups["fine"], key=lambda x: (x[1]["start_time"], x[1]["end_time"]))
            mids = sorted(level_groups["mid"], key=lambda x: (x[1]["start_time"], x[1]["end_time"]))
            coarses = sorted(level_groups["coarse"], key=lambda x: (x[1]["start_time"], x[1]["end_time"]))

            if not fines or not mids or not coarses:
                continue

            for coarse_idx, coarse_sample in coarses:
                chosen_mid = None
                for mid_idx, mid_sample in mids:
                    if (
                        coarse_sample["start_time"] <= mid_sample["start_time"]
                        and mid_sample["end_time"] <= coarse_sample["end_time"]
                    ):
                        chosen_mid = (mid_idx, mid_sample)
                        break

                if chosen_mid is None:
                    continue

                mid_idx, mid_sample = chosen_mid
                chosen_fine = None
                for fine_idx, fine_sample in fines:
                    if (
                        mid_sample["start_time"] <= fine_sample["start_time"]
                        and fine_sample["end_time"] <= mid_sample["end_time"]
                    ):
                        chosen_fine = (fine_idx, fine_sample)
                        break

                if chosen_fine is None:
                    continue

                fine_idx, _ = chosen_fine
                triplets.append((fine_idx, mid_idx, coarse_idx))

        return triplets

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

        triplet_pool = []
        if self.require_complete_triplet:
            allowed_fines = set(level_pools.get("fine", []))
            allowed_mids = set(level_pools.get("mid", []))
            allowed_coarses = set(level_pools.get("coarse", []))
            triplets = [
                triplet
                for triplet in self.complete_triplets
                if (
                    triplet[0] in allowed_fines
                    and triplet[1] in allowed_mids
                    and triplet[2] in allowed_coarses
                )
            ]
            triplet_rng = random.Random(self.seed + self.epoch * 1000 + 999)
            if self.shuffle:
                triplet_rng.shuffle(triplets)
            triplet_pool = triplets
            if not triplet_pool:
                fallback_triplets = list(self.complete_triplets[self.rank::self.num_replicas])
                if self.shuffle:
                    triplet_rng.shuffle(fallback_triplets)
                triplet_pool = fallback_triplets or list(self.complete_triplets)
            if not triplet_pool:
                raise RuntimeError(
                    f"Rank {self.rank} could not build any complete triplet anchors."
                )

        def draw_from_level(level, n, forbidden=None):
            if n <= 0:
                return []

            pool = level_pools[level]
            ptr = level_ptrs[level]
            forbidden = set() if forbidden is None else set(forbidden)

            available = sum(1 for idx in pool if idx not in forbidden)
            if available < n and self.drop_last:
                raise RuntimeError(
                    f"Not enough non-anchor samples for level={level} on rank={self.rank}. "
                    f"Need {n}, have {available}."
                )

            if len(pool) < n and self.drop_last:
                raise RuntimeError(
                    f"Not enough samples for level={level} on rank={self.rank}. "
                    f"Need {n}, have {len(pool)}."
                )

            out = []
            seen = set(forbidden)
            while len(out) < n:
                if ptr >= len(pool):
                    ptr = 0
                    if self.shuffle and len(pool) > 1:
                        level_rngs[level].shuffle(pool)

                candidate = pool[ptr]
                ptr += 1
                if candidate in seen:
                    continue
                out.append(candidate)
                seen.add(candidate)

            level_ptrs[level] = ptr
            return out

        batch_rng = random.Random(self.seed + self.epoch * 100000 + self.rank)

        for batch_idx in range(num_batches):
            batch = []
            anchors = []
            if self.require_complete_triplet:
                anchors = list(triplet_pool[batch_idx % len(triplet_pool)])
                batch.extend(anchors)

            for level in self.level_order:
                take_n = self.level_batch_sizes[level]
                if take_n > 0:
                    anchor_count = 0
                    if self.require_complete_triplet and level in ("fine", "mid", "coarse"):
                        anchor_count = 1
                    batch.extend(draw_from_level(level, take_n - anchor_count, forbidden=batch))

            if self.shuffle:
                if anchors:
                    tail = batch[len(anchors):]
                    batch_rng.shuffle(tail)
                    batch = anchors + tail
                else:
                    batch_rng.shuffle(batch)

            yield batch
