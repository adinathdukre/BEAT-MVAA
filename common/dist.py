from __future__ import annotations

import itertools
import os
import random

import torch
import torch.distributed as dist


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if world_size > 1 else None
    if world_size > 1:
        n_visible = torch.cuda.device_count()
        if n_visible > 1 and world_size > n_visible:
            raise RuntimeError(
                f"[dist] over-subscription: {world_size} ranks but only {n_visible} GPUs visible "
                f"(rank {os.environ.get('RANK')}, local_rank {local_rank}). Each rank needs its own GPU "
                f"- launch torchrun with --nproc_per_node <= visible GPUs (e.g. NGPU={n_visible}).")
        dev_idx = local_rank if n_visible > 1 else 0
        torch.cuda.set_device(dev_idx)
        print(f"[dist] rank={os.environ.get('RANK', 0)} local_rank={local_rank} -> cuda:{dev_idx} "
              f"(visible={n_visible}) master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
              flush=True)
        dist.init_process_group(backend="nccl", init_method="env://")
        return dist.get_rank(), world_size, torch.device("cuda", dev_idx)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def barrier() -> None:
    if is_dist():
        dist.barrier()


def cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def reduce_mean(value: float, device) -> float:
    if not is_dist():
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size())


def wrap_ddp(model, device, find_unused_parameters: bool = False):
    if is_dist():
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=find_unused_parameters,
        )
    return model


def _fsdp_bf16():
    from torch.distributed.fsdp import MixedPrecision
    return MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                          buffer_dtype=torch.bfloat16)


def wrap_fsdp(module, device, min_num_params: int = 10_000_000, mixed_precision: bool = True):
    import functools

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    return FSDP(
        module,
        auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=_fsdp_bf16() if mixed_precision else None,
        device_id=device,
        sync_module_states=True,
        use_orig_params=False,
    )


def fsdp_full_state_dict(module):
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        return module.state_dict()


def make_loader(dataset, batch_size, shuffle, num_workers, collate_fn=None,
                drop_last=False, pin_memory=False, batch_sampler=None):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if batch_sampler is not None:
        loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=collate_fn)
        return loader, batch_sampler

    sampler = DistributedSampler(dataset, shuffle=shuffle) if is_dist() else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None),
                        sampler=sampler, num_workers=num_workers, drop_last=drop_last,
                        pin_memory=pin_memory, collate_fn=collate_fn)
    return loader, sampler


class FixedStepBatchSampler(torch.utils.data.Sampler):
    def __init__(self, indices, batch_size, num_batches, rank=0, world_size=1, seed=0):
        self.indices = list(indices)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.rank = int(rank)
        self.world = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_batch = 0
        if not self.indices:
            raise ValueError("fixed-step sampling requires at least one index")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if self.world <= 0:
            raise ValueError("world_size must be positive")
        if self.rank < 0 or self.rank >= self.world:
            raise ValueError("rank must satisfy 0 <= rank < world_size")

    def set_epoch(self, epoch, start_batch=0):
        epoch = int(epoch)
        start_batch = int(start_batch)
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if start_batch < 0 or start_batch > self.num_batches:
            raise ValueError("start_batch must be within the epoch")
        self.epoch = epoch
        self.start_batch = start_batch

    def state_dict(self):
        return {
            "schema_version": "fixed-step-batch-sampler-v1",
            "indices": list(self.indices),
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "rank": self.rank,
            "world_size": self.world,
            "seed": self.seed,
            "epoch": self.epoch,
            "start_batch": self.start_batch,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in (
            "schema_version",
            "indices",
            "batch_size",
            "num_batches",
            "rank",
            "world_size",
            "seed",
        ):
            if state.get(key) != expected[key]:
                raise ValueError(f"fixed-step sampler state mismatch for {key}")
        self.set_epoch(state["epoch"], state.get("start_batch", 0))

    def _global_stream(self):
        required = self.num_batches * self.batch_size * self.world
        rng = random.Random(self.seed + self.epoch * 1000003)
        stream = []
        while len(stream) < required:
            permutation = list(self.indices)
            rng.shuffle(permutation)
            stream.extend(permutation)
        return stream[:required]

    def __iter__(self):
        stream = self._global_stream()
        global_batch_size = self.batch_size * self.world
        local_start = self.rank * self.batch_size
        for batch_index in range(self.start_batch, self.num_batches):
            offset = batch_index * global_batch_size + local_start
            yield stream[offset:offset + self.batch_size]

    def __len__(self):
        return self.num_batches - self.start_batch


class TwoStreamBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labeled_idx, unlabeled_idx, batch_size, labeled_fraction=0.5,
                 rank=0, world_size=1, seed=0, max_batches=0):
        self.labeled = list(labeled_idx)
        self.unlabeled = list(unlabeled_idx)
        self.rank = rank
        self.world = max(1, world_size)
        self.seed = seed
        self.epoch = 0
        self.labeled_fraction = float(labeled_fraction)
        self.singleton_mixed = batch_size == 1 and bool(self.labeled) and bool(self.unlabeled)
        if self.singleton_mixed:
            lb = 0
        elif self.labeled and self.unlabeled:
            lb = min(max(round(batch_size * labeled_fraction), 1), batch_size - 1)
        elif self.labeled:
            lb = batch_size
        else:
            lb = 0
        self.labeled_bs = lb
        self.unlabeled_bs = batch_size - lb
        self._primary, self._primary_bs = (
            (self.unlabeled, self.unlabeled_bs) if self.unlabeled_bs > 0
            else (self.labeled, self.labeled_bs))
        per_rank = len(self._primary) // self.world
        self._n = per_rank // max(1, self._primary_bs) if per_rank else 0
        if max_batches and max_batches > 0:
            self._n = min(self._n, int(max_batches))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _shuffled(self, items, salt):
        g = random.Random(self.seed + self.epoch * 100003 + salt)
        out = list(items)
        g.shuffle(out)
        return out

    def __iter__(self):
        if self._n == 0:
            return
        primary = self._shuffled(self._primary, 1)[self.rank::self.world]
        if self.singleton_mixed:
            n_labeled = int(round(self._n * self.labeled_fraction))
            schedule = [True] * n_labeled + [False] * (self._n - n_labeled)
            random.Random(self.seed + self.epoch * 100003 + 17).shuffle(schedule)
            labeled = itertools.cycle(self._shuffled(self.labeled, 7 + self.rank))
            unlabeled = iter(primary)
            for use_labeled in schedule:
                yield [next(labeled) if use_labeled else next(unlabeled)]
            return
        if self.labeled_bs and self.unlabeled_bs:
            cyc = itertools.cycle(self._shuffled(self.labeled, 7 + self.rank))
            for b in range(self._n):
                u = primary[b * self.unlabeled_bs:(b + 1) * self.unlabeled_bs]
                yield [next(cyc) for _ in range(self.labeled_bs)] + u
        else:
            bs = self._primary_bs
            for b in range(self._n):
                yield primary[b * bs:(b + 1) * bs]

    def __len__(self):
        return self._n
