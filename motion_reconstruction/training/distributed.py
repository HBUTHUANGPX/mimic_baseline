"""分布式训练辅助。

本模块只处理单节点 `torch.distributed.run` / `torchrun` 场景下的
进程组初始化、rank 信息和常用 collective，尽量不把训练逻辑揉进去。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Mapping

import torch
import torch.distributed as dist


@dataclass
class DistributedConfigView:
    """训练配置里与分布式相关的只读视图。"""

    enabled: bool | str = "auto"
    backend: str | None = None
    find_unused_parameters: bool = False
    shard_strategy: str = "valid_centers_greedy"
    timeout_minutes: int = 30


@dataclass
class DistributedRuntime:
    """当前进程的分布式运行时信息。"""

    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str | None = None
    initialized_by_me: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_multi_process(self) -> bool:
        return self.world_size > 1

    @classmethod
    def from_env(
        cls,
        *,
        requested_device: str,
        config: DistributedConfigView | None = None,
    ) -> "DistributedRuntime":
        view = config or DistributedConfigView()
        has_rank_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        should_enable = _should_enable(view.enabled, has_rank_env)
        if not should_enable:
            return cls(enabled=False)

        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        backend = view.backend or _default_backend(requested_device)

        initialized_by_me = False
        if not dist.is_initialized():
            init_kwargs = {}
            if backend == "nccl":
                torch.cuda.set_device(local_rank)
                init_kwargs["device_id"] = torch.device("cuda", local_rank)
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=int(view.timeout_minutes)),
                **init_kwargs,
            )
            initialized_by_me = True
        else:
            backend = dist.get_backend()
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        return cls(
            enabled=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=backend,
            initialized_by_me=initialized_by_me,
        )

    def broadcast_string(self, value: str | None, *, src: int = 0) -> str | None:
        if not self.enabled:
            return value
        payload = [value]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]

    def barrier(self) -> None:
        if self.enabled:
            if self.backend == "nccl":
                dist.barrier(device_ids=[torch.cuda.current_device()])
                return
            dist.barrier()

    def reduce_mean_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return value
        tensor = value.detach().clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / float(self.world_size)

    def reduce_dict_sum(self, values: Mapping[str, float]) -> dict[str, float]:
        if not values:
            return {}
        keys = sorted(values)
        tensor = torch.tensor(
            [float(values[key]) for key in keys],
            dtype=torch.float32,
            device=self.collective_device,
        )
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return {key: float(tensor[index].item()) for index, key in enumerate(keys)}

    def reduce_max_int(self, value: int) -> int:
        tensor = torch.tensor([int(value)], dtype=torch.long, device=self.collective_device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    def reduce_max_float(self, value: float) -> float:
        tensor = torch.tensor([float(value)], dtype=torch.float32, device=self.collective_device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def reduce_statistics(
        self,
        *,
        count: int,
        feature_sum: torch.Tensor,
        feature_sumsq: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return int(count), feature_sum, feature_sumsq
        count_tensor = torch.tensor([int(count)], dtype=torch.long, device=self.collective_device)
        sum_tensor = feature_sum.detach().clone()
        sumsq_tensor = feature_sumsq.detach().clone()
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sumsq_tensor, op=dist.ReduceOp.SUM)
        return int(count_tensor.item()), sum_tensor, sumsq_tensor

    @property
    def collective_device(self) -> torch.device:
        if self.backend == "nccl" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def close(self) -> None:
        if self.enabled and self.initialized_by_me and dist.is_initialized():
            dist.destroy_process_group()


def distributed_config_from_object(config) -> DistributedConfigView:
    """把 dataclass 或 duck-typed 配置对象转成运行时视图。"""
    if config is None:
        return DistributedConfigView()
    return DistributedConfigView(
        enabled=getattr(config, "enabled", "auto"),
        backend=getattr(config, "backend", None),
        find_unused_parameters=bool(getattr(config, "find_unused_parameters", False)),
        shard_strategy=str(getattr(config, "shard_strategy", "valid_centers_greedy")),
        timeout_minutes=int(getattr(config, "timeout_minutes", 30)),
    )


def _should_enable(enabled: bool | str, has_rank_env: bool) -> bool:
    if isinstance(enabled, str):
        if enabled.lower() == "auto":
            return has_rank_env
        if enabled.lower() in {"true", "1", "yes", "on"}:
            return True
        if enabled.lower() in {"false", "0", "no", "off"}:
            return False
    return bool(enabled)


def _default_backend(requested_device: str) -> str:
    requested = requested_device.lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return "nccl"
    return "gloo"
