from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import torch


DEFAULT_FPS = 50
DEFAULT_HOURS = 300
DEFAULT_TOTAL_FRAMES = DEFAULT_FPS * 60 * 60 * DEFAULT_HOURS
DEFAULT_NUM_ENVS = 4096


@dataclass(frozen=True)
class DifficultyDatasetConfig:
    total_frames: int = DEFAULT_TOTAL_FRAMES
    max_window_span: int = 2
    device: str = "cpu"


@dataclass(frozen=True)
class DifficultyDataset:
    difficulty: torch.Tensor

    @classmethod
    def generate(cls, config: DifficultyDatasetConfig) -> "DifficultyDataset":
        if config.total_frames <= 0:
            raise ValueError("total_frames must be positive.")
        if config.max_window_span < 0:
            raise ValueError("max_window_span must be non-negative.")

        counts = _quadratic_difficulty_counts(config.total_frames)
        chunks = [
            torch.full((int(counts[d - 1].item()),), d, dtype=torch.uint8)
            for d in range(1, 11)
        ]
        difficulty = torch.cat(chunks).to(config.device)
        dataset = cls(difficulty=difficulty)
        _validate_sorted_counts_window_constraint(
            counts=counts,
            max_window_span=config.max_window_span,
        )
        return dataset

    @property
    def total_frames(self) -> int:
        return int(self.difficulty.numel())

    @property
    def device(self) -> torch.device:
        return self.difficulty.device

    def required_sample_count(self) -> torch.Tensor:
        difficulty = self.difficulty.to(torch.int32)
        return difficulty.square() * 20

    def validate(self, max_window_span: int = 2) -> None:
        if self.difficulty.ndim != 1:
            raise ValueError("difficulty must be a 1-D tensor.")
        if self.total_frames == 0:
            raise ValueError("difficulty must contain at least one frame.")
        min_difficulty = int(self.difficulty.min().item())
        max_difficulty = int(self.difficulty.max().item())
        if min_difficulty < 1 or max_difficulty > 10:
            raise ValueError("difficulty levels must be in [1, 10].")
        if self.total_frames >= 10:
            windows = self.difficulty.unfold(0, 10, 1).to(torch.int16)
            spans = windows.max(dim=1).values - windows.min(dim=1).values
            if int(spans.max().item()) > max_window_span:
                raise ValueError(
                    "At least one 10-frame window violates max_window_span."
                )


class LearningState:
    def __init__(self, dataset: DifficultyDataset) -> None:
        self.dataset = dataset
        self.sample_counts = torch.zeros(
            dataset.total_frames,
            dtype=torch.int32,
            device=dataset.device,
        )
        self.required_counts = dataset.required_sample_count()
        self.learned_mask = torch.zeros(
            dataset.total_frames,
            dtype=torch.bool,
            device=dataset.device,
        )

    def record_samples(self, frame_ids: torch.Tensor) -> torch.Tensor:
        frame_ids = frame_ids.to(device=self.dataset.device, dtype=torch.long)
        failed_before_update = ~self.learned_mask[frame_ids]
        unique_frame_ids, increments = torch.unique(frame_ids, return_counts=True)
        self.sample_counts[unique_frame_ids] += increments.to(self.sample_counts.dtype)
        self.learned_mask[unique_frame_ids] = (
            self.sample_counts[unique_frame_ids]
            >= self.required_counts[unique_frame_ids]
        )
        return failed_before_update

    @property
    def learned_frames(self) -> int:
        return int(self.learned_mask.sum().item())

    @property
    def learned_ratio(self) -> float:
        return self.learned_frames / float(self.dataset.total_frames)


@dataclass(frozen=True)
class SimulationConfig:
    num_envs: int = DEFAULT_NUM_ENVS
    max_iterations: int = 1_000
    target_learned_ratio: float = 1.0


@dataclass(frozen=True)
class SimulationResult:
    sampler_name: str
    iterations: int
    samples: int
    learned_frames: int
    total_frames: int
    learned_ratio: float
    mean_samples_per_frame: float
    learned_frames_by_difficulty: dict[int, int]
    total_frames_by_difficulty: dict[int, int]


class SamplerAdapter(ABC):
    name: str

    def __init__(self, num_envs: int, dataset: DifficultyDataset) -> None:
        self.num_envs = num_envs
        self.dataset = dataset

    @abstractmethod
    def sample_frame_ids(self) -> torch.Tensor:
        pass

    @abstractmethod
    def observe(
        self,
        frame_ids: torch.Tensor,
        failures: torch.Tensor,
        state: LearningState,
    ) -> None:
        pass


class UniformSamplerAdapter(SamplerAdapter):
    name = "uniform"

    def __init__(
        self,
        num_envs: int,
        dataset: DifficultyDataset,
        seed: int = 1,
    ) -> None:
        super().__init__(num_envs=num_envs, dataset=dataset)
        self.generator = torch.Generator(device=dataset.device)
        self.generator.manual_seed(seed)

    def sample_frame_ids(self) -> torch.Tensor:
        return torch.randint(
            self.dataset.total_frames,
            (self.num_envs,),
            generator=self.generator,
            device=self.dataset.device,
        )

    def observe(
        self,
        frame_ids: torch.Tensor,
        failures: torch.Tensor,
        state: LearningState,
    ) -> None:
        del frame_ids, failures, state


class _AdaptiveSamplerAdapter(SamplerAdapter):
    sampler_cfg_class_name: str

    def __init__(
        self,
        num_envs: int,
        dataset: DifficultyDataset,
        bin_frame_count: int = DEFAULT_FPS,
        seed: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__(num_envs=num_envs, dataset=dataset)
        if bin_frame_count <= 0:
            raise ValueError("bin_frame_count must be positive.")
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.command = _MockCommand(
            num_envs=num_envs,
            total_frames=dataset.total_frames,
            bin_frame_count=bin_frame_count,
            device=self.device,
        )
        adaptive_sample = _load_adaptive_sample_module()
        cfg_class = getattr(adaptive_sample, self.sampler_cfg_class_name)
        self.sampler = cfg_class.class_type(self.command, cfg_class())

    def sample_frame_ids(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.sampler.on_resample_start(env_ids, update_failure_statistics=True)
        probabilities = self.sampler.build_sampling_probabilities()
        sampled_bins = torch.multinomial(
            probabilities,
            self.num_envs,
            replacement=True,
            generator=self.generator,
        )
        frame_ids = self._sample_frames_from_bins(sampled_bins)
        self.command.time_steps = frame_ids
        self.sampler.on_resample_complete(
            env_ids,
            sampled_bins,
            update_failure_statistics=True,
        )
        return frame_ids.to(self.dataset.device)

    def observe(
        self,
        frame_ids: torch.Tensor,
        failures: torch.Tensor,
        state: LearningState,
    ) -> None:
        del state
        self.command._previous_time_steps = frame_ids.to(
            device=self.device,
            dtype=torch.long,
        )
        self.command._env.termination_manager.terminated = failures.to(
            device=self.device,
            dtype=torch.bool,
        )
        self.sampler.on_step_end()

    def _sample_frames_from_bins(self, sampled_bins: torch.Tensor) -> torch.Tensor:
        offsets = torch.randint(
            self.command.bin_frame_count,
            sampled_bins.shape,
            generator=self.generator,
            device=self.device,
        )
        frame_ids = sampled_bins * self.command.bin_frame_count + offsets
        return torch.clamp(frame_ids, max=self.dataset.total_frames - 1)


class LegacySamplerAdapter(_AdaptiveSamplerAdapter):
    name = "legacy"
    sampler_cfg_class_name = "LegacyBinAdaptiveSamplingCfg"


class StratifiedLegacySamplerAdapter(_AdaptiveSamplerAdapter):
    name = "stratified_legacy"
    sampler_cfg_class_name = "StratifiedLegacyBinAdaptiveSamplingCfg"


class SonicSamplerAdapter(_AdaptiveSamplerAdapter):
    name = "sonic"
    sampler_cfg_class_name = "SonicBinAdaptiveSamplingCfg"


class TrainingSimulator:
    def __init__(
        self,
        dataset: DifficultyDataset,
        sampler: SamplerAdapter,
        config: SimulationConfig,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.config = config
        self.state = LearningState(dataset)

    def run(self) -> SimulationResult:
        iterations = 0
        for iteration in range(1, self.config.max_iterations + 1):
            frame_ids = self.sampler.sample_frame_ids()
            failures = self.state.record_samples(frame_ids)
            self.sampler.observe(frame_ids, failures, self.state)
            iterations = iteration
            if self.state.learned_ratio >= self.config.target_learned_ratio:
                break

        samples = iterations * self.config.num_envs
        return SimulationResult(
            sampler_name=self.sampler.name,
            iterations=iterations,
            samples=samples,
            learned_frames=self.state.learned_frames,
            total_frames=self.dataset.total_frames,
            learned_ratio=self.state.learned_ratio,
            mean_samples_per_frame=float(self.state.sample_counts.float().mean().item()),
            learned_frames_by_difficulty=self._count_by_difficulty(
                self.state.learned_mask
            ),
            total_frames_by_difficulty=self._count_by_difficulty(
                torch.ones_like(self.state.learned_mask)
            ),
        )

    def _count_by_difficulty(self, mask: torch.Tensor) -> dict[int, int]:
        counts: dict[int, int] = {}
        for difficulty in range(1, 11):
            difficulty_mask = self.dataset.difficulty == difficulty
            counts[difficulty] = int((difficulty_mask & mask).sum().item())
        return counts


@dataclass
class _MockTerminationManager:
    terminated: torch.Tensor


@dataclass
class _MockEnv:
    termination_manager: _MockTerminationManager


class _MockCommand:
    def __init__(
        self,
        num_envs: int,
        total_frames: int,
        bin_frame_count: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.bin_frame_count = bin_frame_count
        self.bin_count = (total_frames + bin_frame_count - 1) // bin_frame_count
        self.valid_sampling_bin_mask = torch.ones(
            self.bin_count,
            dtype=torch.bool,
            device=device,
        )
        self.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._previous_time_steps = None
        self._env = _MockEnv(
            termination_manager=_MockTerminationManager(
                terminated=torch.zeros(num_envs, dtype=torch.bool, device=device)
            )
        )


def _quadratic_difficulty_counts(total_frames: int) -> torch.Tensor:
    weights = torch.tensor([(11 - d) ** 2 for d in range(1, 11)], dtype=torch.float64)
    exact = weights * float(total_frames) / float(weights.sum().item())
    counts = torch.floor(exact).to(torch.long)
    remainder = total_frames - int(counts.sum().item())
    if remainder > 0:
        fractional_order = torch.argsort(exact - counts.to(exact.dtype), descending=True)
        counts[fractional_order[:remainder]] += 1
    return counts


def _validate_sorted_counts_window_constraint(
    counts: torch.Tensor,
    max_window_span: int,
) -> None:
    boundary_probe = []
    for difficulty, count in enumerate(counts.tolist(), start=1):
        boundary_probe.extend([difficulty] * min(int(count), 9))
    if len(boundary_probe) < 10:
        return
    probe = torch.tensor(boundary_probe, dtype=torch.uint8)
    windows = probe.unfold(0, 10, 1).to(torch.int16)
    spans = windows.max(dim=1).values - windows.min(dim=1).values
    if int(spans.max().item()) > max_window_span:
        raise ValueError(
            "Generated difficulty sequence violates the 10-frame window constraint."
        )


def _load_adaptive_sample_module():
    module_name = "_adaptive_sample_for_sampling_experiment"
    if module_name in sys.modules:
        return sys.modules[module_name]

    adaptive_sample_path = (
        Path(__file__).resolve().parents[1]
        / "tasks"
        / "tracking"
        / "mdp"
        / "adaptive_sample.py"
    )
    previous_modules = {
        name: sys.modules.get(name)
        for name in ("isaaclab", "isaaclab.managers", "isaaclab.utils")
    }
    _install_isaaclab_stubs()
    try:
        spec = importlib.util.spec_from_file_location(module_name, adaptive_sample_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {adaptive_sample_path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def _install_isaaclab_stubs() -> None:
    isaaclab_module = types.ModuleType("isaaclab")
    managers_module = types.ModuleType("isaaclab.managers")
    utils_module = types.ModuleType("isaaclab.utils")

    class CommandTerm:
        pass

    managers_module.CommandTerm = CommandTerm
    utils_module.configclass = dataclass
    isaaclab_module.managers = managers_module
    isaaclab_module.utils = utils_module
    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.managers"] = managers_module
    sys.modules["isaaclab.utils"] = utils_module
