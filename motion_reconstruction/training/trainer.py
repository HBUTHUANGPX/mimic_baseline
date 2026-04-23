"""动作重构的端到端训练器。

训练器是薄编排层：它把配置连接到来源解析器、raw loader、FeatureBuilder、
GPU 窗口缓冲、模型、损失、checkpoint 和 TensorBoard。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from motion_reconstruction.config.schema import MotionReconstructionConfig
from motion_reconstruction.data import build_motion_shard_plan
from motion_reconstruction.pipeline import (
    MotionRuntimeBundle,
    ResolvedMotionFiles,
    build_autoencoder,
    build_motion_runtime,
    resolve_motion_files,
)
from motion_reconstruction.training.checkpoint import save_checkpoint
from motion_reconstruction.training.distributed import DistributedRuntime, distributed_config_from_object
from motion_reconstruction.training.losses import DualReconstructionLoss, LossOutput
from motion_reconstruction.training.normalization import WindowFeatureNormalizer


class NullSummaryWriter:
    """TensorBoard 不可用或当前不是主进程时使用的空写入器。"""

    def add_scalar(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def close(self):
        return None


class MotionReconstructionTrainer:
    """从配置构建数据和模型状态，并训练重构模型。"""

    def __init__(self, config: MotionReconstructionConfig):
        self.config = config
        self.distributed = DistributedRuntime.from_env(
            requested_device=config.train.device,
            config=distributed_config_from_object(config.train.distributed),
        )
        self.device = self._resolve_device(config.train.device)
        self.global_step = 0

        torch.manual_seed(config.train.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.train.seed)

        self.run_dir = self._make_run_dir()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "tb"
        self.writer = self._make_writer(self.log_dir) if self.distributed.is_main_process else NullSummaryWriter()
        self._emit(f"输出目录: {self.run_dir}")
        self._emit(f"训练设备: {self.device}")
        if self.distributed.enabled:
            self._emit(
                "分布式运行时: "
                f"backend={self.distributed.backend}, world_size={self.distributed.world_size}, "
                f"local_rank={self.distributed.local_rank}"
            )
        self._emit("准备数据...")

        self.runtime, self.normalizers = self._build_data()
        self.features = self.runtime.features
        self.buffer = self.runtime.buffer
        self.batch_total = self._num_batches()

        window_size = self.buffer.window_size
        robot_input_dim = self.runtime.robot_input_dim
        human_input_dim = self.runtime.human_input_dim
        self.model, self.quantizer_config = build_autoencoder(
            self.config,
            robot_input_dim=robot_input_dim,
            human_input_dim=human_input_dim,
        )
        self.model = self.model.to(self.device)
        if self.distributed.enabled:
            ddp_kwargs = {
                "find_unused_parameters": bool(self.config.train.distributed.find_unused_parameters),
            }
            if self.device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                )
            self.model = DistributedDataParallel(self.model, **ddp_kwargs)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        self.loss_fn = DualReconstructionLoss(**self.config.loss.__dict__)
        self._emit(
            "模型维度: "
            f"robot_input={robot_input_dim}, human_input={human_input_dim}, "
            f"latent={self.config.model.latent_dim}, window={window_size}"
        )

    def train(self) -> None:
        """按 epoch 训练，每个 epoch 对完整合法中心帧池随机打乱。"""
        generator = self._make_generator()
        generator.manual_seed(self.config.train.seed)
        self._emit(
            "开始训练: "
            f"epochs={self.config.train.epochs}, batch_size={self.config.train.batch_size}, "
            f"每个 epoch 的 batch 数={self.batch_total}"
        )

        try:
            epoch_iter = range(1, self.config.train.epochs + 1)
            epoch_bar = self._make_progress(
                epoch_iter,
                total=self.config.train.epochs,
                desc="训练 epoch",
                unit="epoch",
            )
            for epoch in epoch_bar:
                epoch_start = time.time()
                totals: dict[str, float] = {"total": 0.0}
                batch_count = 0
                self.model.train()

                batch_iter = self.buffer.iter_epoch_batches(
                    self.config.train.batch_size,
                    generator=generator,
                    num_batches=self.batch_total if self.distributed.enabled else None,
                )
                batch_bar = self._make_progress(
                    batch_iter,
                    total=self.batch_total,
                    desc=f"epoch {epoch}/{self.config.train.epochs}",
                    unit="batch",
                    leave=False,
                )
                for batch in batch_bar:
                    robot = batch.robot_window.reshape(batch.robot_window.shape[0], -1)
                    human = batch.human_window.reshape(batch.human_window.shape[0], -1)
                    robot_norm = self.normalizers["robot"](robot)
                    human_norm = self.normalizers["human"](human)

                    output = self.model(robot_norm, human_norm)
                    loss_output = self.loss_fn(output, robot_norm)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss_output.total.backward()
                    self.optimizer.step()

                    batch_count += 1
                    self.global_step += 1
                    totals["total"] += float(loss_output.total.detach().cpu())
                    for name, value in loss_output.terms.items():
                        totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())

                    if self.global_step % self.config.train.log_every_steps == 0:
                        loss_values, quantizer_values = self._reduce_step_metrics(loss_output, output)
                        if self.distributed.is_main_process:
                            self._log_step(loss_values, quantizer_values, output)
                            self._set_progress_postfix(batch_bar, total=loss_values["total"])

                epoch_time = self.distributed.reduce_max_float(time.time() - epoch_start)
                reduced_totals = self.distributed.reduce_dict_sum(totals)
                global_batch_count = batch_count * self.distributed.world_size
                if self.distributed.is_main_process:
                    self._log_epoch(epoch, reduced_totals, global_batch_count, epoch_time)
                    self._save(epoch, "latest.pt")
                    if (
                        self.config.train.checkpoint_interval_epochs > 0
                        and epoch % self.config.train.checkpoint_interval_epochs == 0
                    ):
                        self._save(epoch, f"epoch_{epoch:04d}.pt")
                    avg_total = reduced_totals["total"] / max(global_batch_count, 1)
                    self._set_progress_postfix(epoch_bar, total=avg_total, step=self.global_step)
                    if not self.config.train.progress:
                        self._emit(
                            f"epoch {epoch}/{self.config.train.epochs}: "
                            f"total={avg_total:.6f}, batches={global_batch_count}, time={epoch_time:.2f}s"
                        )
                self.distributed.barrier()
        finally:
            self.writer.close()
            self.distributed.close()

        self._emit(f"训练结束，latest checkpoint: {self.ckpt_dir / 'latest.pt'}")

    def _build_data(self) -> tuple[MotionRuntimeBundle, dict[str, WindowFeatureNormalizer]]:
        """加载 raw motion、构建特征、创建窗口缓冲并拟合归一化器。"""
        resolved = resolve_motion_files(self.config)
        scan_progress = self.config.train.progress and self.distributed.is_main_process
        shard_plan = build_motion_shard_plan(
            files=resolved.paths,
            groups=resolved.groups,
            history=self.config.train.history,
            future=self.config.train.future,
            world_size=self.distributed.world_size,
            progress=scan_progress,
        )
        self.shard_plan = shard_plan
        local_shard = shard_plan.shards[self.distributed.rank]
        self._emit(
            "数据分片: "
            f"总文件={len(resolved.paths)}, 总合法中心帧={shard_plan.total_valid_centers}, "
            f"各 rank 合法中心帧={[item.valid_center_count for item in shard_plan.shards]}"
        )
        local_resolved = ResolvedMotionFiles(paths=local_shard.paths, groups=local_shard.groups)
        runtime = build_motion_runtime(
            self.config,
            device=self.device,
            emit=self._emit,
            resolved=local_resolved,
            progress=self.config.train.progress and self.distributed.is_main_process,
        )
        self._emit(
            "当前 rank 数据: "
            f"clips={len(local_shard.paths)}, frames={local_shard.frame_count}, "
            f"合法中心帧={local_shard.valid_center_count}"
        )
        normalizers = {
            "robot": self._fit_normalizer(runtime.features.robot, runtime.window_size),
            "human": self._fit_normalizer(runtime.features.human, runtime.window_size),
        }
        return runtime, normalizers

    def _fit_normalizer(self, frame_features: torch.Tensor, window_size: int) -> WindowFeatureNormalizer:
        count, feature_sum, feature_sumsq = WindowFeatureNormalizer.compute_statistics(frame_features)
        count, feature_sum, feature_sumsq = self.distributed.reduce_statistics(
            count=count,
            feature_sum=feature_sum,
            feature_sumsq=feature_sumsq,
        )
        return WindowFeatureNormalizer.from_statistics(
            count=count,
            feature_sum=feature_sum,
            feature_sumsq=feature_sumsq,
            window_size=window_size,
            eps=self.config.train.normalizer_eps,
        ).to(self.device)

    def _reduce_step_metrics(self, loss_output: LossOutput, output) -> tuple[dict[str, float], dict[str, float]]:
        loss_values = {"total": float(loss_output.total.detach().cpu())}
        loss_values.update({name: float(value.detach().cpu()) for name, value in loss_output.terms.items()})
        reduced_losses = self._reduce_mean_scalars(loss_values)

        quantizer_values: dict[str, float] = {}
        for name, quantizer_output in (
            ("robot", output.robot_quantizer),
            ("human", output.human_quantizer),
            ("cycle", output.cycle_quantizer),
        ):
            for stat_name, stat_value in quantizer_output.stats.items():
                value = stat_value.item() if torch.is_tensor(stat_value) else float(stat_value)
                quantizer_values[f"{name}_{stat_name}"] = float(value)
        reduced_quantizers = self._reduce_mean_scalars(quantizer_values)
        return reduced_losses, reduced_quantizers

    def _reduce_mean_scalars(self, values: dict[str, float]) -> dict[str, float]:
        reduced = self.distributed.reduce_dict_sum(values)
        if not self.distributed.enabled:
            return reduced
        return {key: value / float(self.distributed.world_size) for key, value in reduced.items()}

    def _log_step(self, loss_values: dict[str, float], quantizer_values: dict[str, float], output) -> None:
        """记录 step 级损失、量化器统计和可选潜变量直方图。"""
        self.writer.add_scalar("train/total_loss", loss_values["total"], self.global_step)
        for name, value in loss_values.items():
            if name == "total":
                continue
            self.writer.add_scalar(f"train/{name}", value, self.global_step)
        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        for name, value in quantizer_values.items():
            self.writer.add_scalar(f"quantizer/{name}", value, self.global_step)
        if self.config.train.log_histograms:
            self.writer.add_histogram("latent/q_robot", output.q_robot.detach().cpu(), self.global_step)
            self.writer.add_histogram("latent/q_human", output.q_human.detach().cpu(), self.global_step)
            self.writer.add_histogram("latent/q_cycle", output.q_cycle.detach().cpu(), self.global_step)

    def _log_epoch(self, epoch: int, totals: dict[str, float], batch_count: int, epoch_time: float) -> None:
        denom = max(batch_count, 1)
        for name, value in totals.items():
            self.writer.add_scalar(f"epoch/{name}", value / denom, epoch)
        self.writer.add_scalar("epoch/time_sec", epoch_time, epoch)

    def _save(self, epoch: int, name: str) -> Path:
        """保存 latest 或周期 checkpoint，并包含全部重构元数据。"""
        return save_checkpoint(
            output_dir=self.ckpt_dir,
            name=name,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            config=self.config.to_dict(),
            normalizers=self.normalizers,
            feature_schema=self.features.schema.to_dict(),
            quantizer_config=self.quantizer_config,
        )

    def _make_run_dir(self) -> Path:
        run_name = self.config.output.run_name
        if run_name is None:
            candidate = datetime.now().strftime("%Y%m%d_%H%M%S") if self.distributed.is_main_process else None
            run_name = self.distributed.broadcast_string(candidate)
        path = Path(self.config.output.root_dir) / str(run_name)
        if self.distributed.is_main_process:
            path.mkdir(parents=True, exist_ok=True)
        self.distributed.barrier()
        return path

    def _num_batches(self) -> int:
        local_batches = self.buffer.num_batches(self.config.train.batch_size)
        return self.distributed.reduce_max_int(local_batches)

    def _make_progress(self, iterable, **kwargs):
        if not self.config.train.progress or not self.distributed.is_main_process:
            return iterable
        return tqdm(iterable, disable=False, dynamic_ncols=True, **kwargs)

    def _set_progress_postfix(self, bar, **values) -> None:
        if not hasattr(bar, "set_postfix"):
            return
        formatted = {
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in values.items()
        }
        bar.set_postfix(formatted)

    def _emit(self, message: str) -> None:
        if not self.config.train.progress or not self.distributed.is_main_process:
            return
        tqdm.write(message)

    def _resolve_device(self, requested: str) -> torch.device:
        requested_lower = requested.lower()
        if requested_lower.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        if requested_lower.startswith("cuda") and self.distributed.enabled:
            return torch.device("cuda", self.distributed.local_rank)
        return torch.device(requested)

    def _make_generator(self) -> torch.Generator:
        return torch.Generator(device=self.device)

    @staticmethod
    def _make_writer(log_dir: Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return NullSummaryWriter()
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir), flush_secs=10)
