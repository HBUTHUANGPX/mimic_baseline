"""动作重构的端到端训练器。

训练器是薄编排层：它把配置连接到来源解析器、raw loader、FeatureBuilder、
GPU 窗口缓冲、模型、损失、checkpoint 和 TensorBoard。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from motion_reconstruction.config.schema import MotionReconstructionConfig
from motion_reconstruction.data import MotionSourceResolver, MotionWindowBuffer, RawMotionLoader
from motion_reconstruction.features import FeatureBuilder, FeatureBuilderConfig
from motion_reconstruction.models import DualFSQAutoEncoder
from motion_reconstruction.models.quantizers import build_quantizer, normalized_quantizer_config
from motion_reconstruction.training.checkpoint import save_checkpoint
from motion_reconstruction.training.losses import DualReconstructionLoss
from motion_reconstruction.training.normalization import WindowFeatureNormalizer


class NullSummaryWriter:
    """TensorBoard 不可用时使用的空写入器。

    缺少 tensorboard 依赖时训练仍可继续，只是不写事件文件。
    """

    def add_scalar(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def close(self):
        return None


class MotionReconstructionTrainer:
    """从配置构建数据和模型状态，并训练重构模型。

    该类适合作为 CLI 的后端，也可以被其它工程直接 import 后调用。
    """

    def __init__(self, config: MotionReconstructionConfig):
        self.config = config
        self.device = self._resolve_device(config.train.device)
        self.global_step = 0
        torch.manual_seed(config.train.seed)

        self.run_dir = self._make_run_dir()
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "tb"
        self.writer = self._make_writer(self.log_dir)
        self._emit(f"输出目录: {self.run_dir}")
        self._emit(f"训练设备: {self.device}")
        self._emit("准备数据...")

        self.features, self.buffer, self.normalizers = self._build_data()
        window_size = self.buffer.window_size
        robot_input_dim = self.features.schema.robot_feature_dim * window_size
        human_input_dim = self.features.schema.human_feature_dim * window_size

        quantizer_config = normalized_quantizer_config(self.config.model.quantizer.__dict__, self.config.model.latent_dim)
        self.quantizer_config = quantizer_config
        quantizer = build_quantizer(quantizer_config, latent_dim=self.config.model.latent_dim)
        self.model = DualFSQAutoEncoder(
            robot_input_dim=robot_input_dim,
            human_input_dim=human_input_dim,
            latent_dim=self.config.model.latent_dim,
            robot_encoder_hidden_dims=self.config.model.robot_encoder_hidden_dims,
            human_encoder_hidden_dims=self.config.model.human_encoder_hidden_dims,
            decoder_hidden_dims=self.config.model.decoder_hidden_dims,
            quantizer=quantizer,
            activation=self.config.model.activation,
        ).to(self.device)
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
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.train.seed)
        batch_total = self._num_batches()
        self._emit(
            "开始训练: "
            f"epochs={self.config.train.epochs}, batch_size={self.config.train.batch_size}, "
            f"每个 epoch 的 batch 数={batch_total}"
        )

        epoch_iter = range(1, self.config.train.epochs + 1)
        epoch_bar = self._make_progress(epoch_iter, total=self.config.train.epochs, desc="训练 epoch", unit="epoch")
        for epoch in epoch_bar:
            epoch_start = time.time()
            totals: dict[str, float] = {"total": 0.0}
            batch_count = 0
            self.model.train()

            batch_iter = self.buffer.iter_epoch_batches(self.config.train.batch_size, generator=generator)
            batch_bar = self._make_progress(
                batch_iter,
                total=batch_total,
                desc=f"epoch {epoch}/{self.config.train.epochs}",
                unit="batch",
                leave=False,
            )
            for batch in batch_bar:
                # 中文：缓冲输出 `[B, W, D]`，MLP 模型输入使用展平窗口 `[B, W*D]`。
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
                    self._log_step(loss_output, output)
                    self._set_progress_postfix(batch_bar, total=float(loss_output.total.detach().cpu()))

            epoch_time = time.time() - epoch_start
            self._log_epoch(epoch, totals, batch_count, epoch_time)
            self._save(epoch, "latest.pt")
            if self.config.train.checkpoint_interval_epochs > 0 and epoch % self.config.train.checkpoint_interval_epochs == 0:
                self._save(epoch, f"epoch_{epoch:04d}.pt")
            avg_total = totals["total"] / max(batch_count, 1)
            self._set_progress_postfix(epoch_bar, total=avg_total, step=self.global_step)
            if not self.config.train.progress:
                self._emit(
                    f"epoch {epoch}/{self.config.train.epochs}: "
                    f"total={avg_total:.6f}, batches={batch_count}, time={epoch_time:.2f}s"
                )

        self.writer.close()
        self._emit(f"训练结束，latest checkpoint: {self.ckpt_dir / 'latest.pt'}")

    def _build_data(self):
        """加载 raw motion、构建特征、创建窗口缓冲并拟合归一化器。"""
        if self.config.data.motion_yaml:
            resolver = MotionSourceResolver.from_legacy_yaml(self.config.data.motion_yaml)
        else:
            resolver = MotionSourceResolver.from_direct_inputs(
                files=self.config.data.files,
                dirs=self.config.data.dirs,
                exclude_files=self.config.data.exclude_files,
                exclude_dirs=self.config.data.exclude_dirs,
            )
        resolved = resolver.resolve(groups=self.config.data.groups or None)
        pairs = resolved.file_group_pairs
        paths = [path for path, _ in pairs]
        groups = [group for _, group in pairs]
        self._emit(f"解析到 motion 文件: {len(paths)}")

        raw = RawMotionLoader(paths, groups=groups).load(device=self.device)
        self._emit(f"加载完成: frames={raw.num_frames}, clips={len(paths)}, fps={raw.fps}")
        feature_builder = FeatureBuilder(
            FeatureBuilderConfig(
                robot_anchor_body=self.config.features.robot_anchor_body,
                human_anchor_body=self.config.features.human_anchor_body,
                human_body_names=self.config.features.human_body_names,
            )
        )
        features = feature_builder.build(raw)
        self._emit(
            "特征维度: "
            f"robot={features.schema.robot_feature_dim}, human={features.schema.human_feature_dim}"
        )
        buffer = MotionWindowBuffer(
            robot_features=features.robot,
            human_features=features.human,
            motion_lengths=raw.motion_lengths,
            history=self.config.train.history,
            future=self.config.train.future,
            device=self.device,
        )
        self._emit(
            "窗口采样: "
            f"history={self.config.train.history}, future={self.config.train.future}, "
            f"window={buffer.window_size}, 合法中心帧={buffer.valid_center_indices.numel()}"
        )
        normalizers = {
            # 中文：robot 归一化器在 robot 编码器输入、解码器输出目标和未来
            # 反归一化中共享。
            "robot": WindowFeatureNormalizer.from_frame_features(
                features.robot,
                window_size=buffer.window_size,
                eps=self.config.train.normalizer_eps,
            ).to(self.device),
            "human": WindowFeatureNormalizer.from_frame_features(
                features.human,
                window_size=buffer.window_size,
                eps=self.config.train.normalizer_eps,
            ).to(self.device),
        }
        return features, buffer, normalizers

    def _log_step(self, loss_output, output) -> None:
        """记录 step 级损失、量化器统计和可选潜变量直方图。"""
        self.writer.add_scalar("train/total_loss", loss_output.total.item(), self.global_step)
        for name, value in loss_output.terms.items():
            self.writer.add_scalar(f"train/{name}", value.item(), self.global_step)
        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        for name, quantizer_output in (
            ("robot", output.robot_quantizer),
            ("human", output.human_quantizer),
            ("cycle", output.cycle_quantizer),
        ):
            for stat_name, stat_value in quantizer_output.stats.items():
                value = stat_value.item() if torch.is_tensor(stat_value) else float(stat_value)
                self.writer.add_scalar(f"quantizer/{name}_{stat_name}", value, self.global_step)
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
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.config.output.root_dir) / run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _num_batches(self) -> int:
        centers = int(self.buffer.valid_center_indices.numel())
        batch_size = int(self.config.train.batch_size)
        return max((centers + batch_size - 1) // batch_size, 1)

    def _make_progress(self, iterable, **kwargs):
        if not self.config.train.progress:
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
        if not self.config.train.progress:
            return
        tqdm.write(message)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    @staticmethod
    def _make_writer(log_dir: Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return NullSummaryWriter()
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir), flush_secs=10)
