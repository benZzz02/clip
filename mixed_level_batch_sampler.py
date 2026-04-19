
import hashlib
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
        anchor_same_video_triplets=False,
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
        self.anchor_same_video_triplets = bool(anchor_same_video_triplets)
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

        self.required_anchor_levels = ("fine", "mid", "coarse")
        self.rank_level_to_indices = None
        self.rank_triplets = None
        if self.anchor_same_video_triplets:
            self._init_same_video_triplets()

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

    def _stable_video_owner(self, video_path):
        digest = hashlib.blake2b(
            f"{self.seed}:{video_path}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, byteorder="big") % self.num_replicas

    def _build_complete_triplets(self):
        samples_by_video = defaultdict(lambda: {"fine": [], "mid": [], "coarse": []})

        for idx, sample in enumerate(self.dataset.samples):
            level = str(sample.get("level", "fine"))
            if level not in self.required_anchor_levels:
                continue
            video_path = sample.get("video_path")
            if not video_path:
                continue
            samples_by_video[video_path][level].append((idx, sample))

        triplets = []
        for video_path, level_groups in samples_by_video.items():
            fines = sorted(
                level_groups["fine"],
                key=lambda x: (float(x[1].get("start_time", 0.0)), float(x[1].get("end_time", 0.0))),
            )
            mids = sorted(
                level_groups["mid"],
                key=lambda x: (float(x[1].get("start_time", 0.0)), float(x[1].get("end_time", 0.0))),
            )
            coarses = sorted(
                level_groups["coarse"],
                key=lambda x: (float(x[1].get("start_time", 0.0)), float(x[1].get("end_time", 0.0))),
            )

            if not fines or not mids or not coarses:
                continue

            for coarse_idx, coarse_sample in coarses:
                coarse_start = float(coarse_sample.get("start_time", 0.0))
                coarse_end = float(coarse_sample.get("end_time", 0.0))

                chosen_mid = None
                for mid_idx, mid_sample in mids:
                    mid_start = float(mid_sample.get("start_time", 0.0))
                    mid_end = float(mid_sample.get("end_time", 0.0))
                    if coarse_start <= mid_start and mid_end <= coarse_end:
                        chosen_mid = (mid_idx, mid_sample)
                        break

                if chosen_mid is None:
                    continue

                mid_idx, mid_sample = chosen_mid
                mid_start = float(mid_sample.get("start_time", 0.0))
                mid_end = float(mid_sample.get("end_time", 0.0))

                chosen_fine = None
                for fine_idx, fine_sample in fines:
                    fine_start = float(fine_sample.get("start_time", 0.0))
                    fine_end = float(fine_sample.get("end_time", 0.0))
                    if mid_start <= fine_start and fine_end <= mid_end:
                        chosen_fine = fine_idx
                        break

                if chosen_fine is None:
                    continue

                triplets.append((video_path, (chosen_fine, mid_idx, coarse_idx)))

        return triplets

    def _init_same_video_triplets(self):
        for level in self.required_anchor_levels:
            if self.level_batch_sizes.get(level, 0) <= 0:
                raise ValueError(
                    "anchor_same_video_triplets=True requires positive batch sizes for "
                    f"{self.required_anchor_levels}."
                )

        self.rank_level_to_indices = {
            rank: defaultdict(list) for rank in range(self.num_replicas)
        }
        for idx, sample in enumerate(self.dataset.samples):
            level = str(sample.get("level", "fine"))
            if level not in self.level_batch_sizes:
                continue
            video_path = sample.get("video_path")
            if not video_path:
                continue
            owner = self._stable_video_owner(video_path)
            self.rank_level_to_indices[owner][level].append(idx)

        self.rank_triplets = {rank: [] for rank in range(self.num_replicas)}
        for video_path, triplet in self._build_complete_triplets():
            owner = self._stable_video_owner(video_path)
            self.rank_triplets[owner].append(triplet)

        min_triplets = min(len(self.rank_triplets[rank]) for rank in range(self.num_replicas))
        if min_triplets == 0:
            raise ValueError(
                "anchor_same_video_triplets=True requires at least one complete "
                "fine-mid-coarse triplet on every rank after video sharding."
            )

    def _num_batches_for_rank(self):
        per_level_batches = []
        for level, take_n in self.level_batch_sizes.items():
            if take_n <= 0:
                continue
            if self.anchor_same_video_triplets:
                rank_level_count = min(
                    len(self.rank_level_to_indices[rank].get(level, []))
                    for rank in range(self.num_replicas)
                )
            else:
                # Use the minimum per-level shard count so every rank can run the same number of batches.
                rank_level_count = self._min_rank_shard_len(len(self.level_to_indices[level]))
            if self.drop_last:
                per_level_batches.append(rank_level_count // take_n)
            else:
                per_level_batches.append(math.ceil(rank_level_count / take_n))

        if self.anchor_same_video_triplets:
            per_level_batches.append(
                min(len(self.rank_triplets[rank]) for rank in range(self.num_replicas))
            )

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
            rng = random.Random(self.seed + self.epoch * 1000 + level_idx)

            if self.anchor_same_video_triplets:
                rank_indices = list(self.rank_level_to_indices[self.rank].get(level, []))
                if self.shuffle and len(rank_indices) > 1:
                    rng.shuffle(rank_indices)
            else:
                indices = list(self.level_to_indices[level])

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
        if self.anchor_same_video_triplets:
            triplets = list(self.rank_triplets[self.rank])
            triplet_rng = random.Random(self.seed + self.epoch * 1000 + 999)
            if self.shuffle:
                triplet_rng.shuffle(triplets)
            triplet_pool = triplets
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
            if self.anchor_same_video_triplets:
                batch.extend(triplet_pool[batch_idx])

            for level in self.level_order:
                take_n = self.level_batch_sizes[level]
                if take_n > 0:
                    anchor_count = 0
                    if self.anchor_same_video_triplets and level in self.required_anchor_levels:
                        anchor_count = 1
                    batch.extend(draw_from_level(level, take_n - anchor_count, forbidden=batch))

            if self.shuffle:
                batch_rng.shuffle(batch)

            yield batch
