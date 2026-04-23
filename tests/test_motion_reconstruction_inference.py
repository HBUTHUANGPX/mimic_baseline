from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_reconstruction.evaluation.reconstruct import ReconstructionResult
from motion_reconstruction.inference.sources import InferenceSourceBundle
from motion_reconstruction.config.schema import MotionReconstructionConfig


class IdentityNormalizer:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


@dataclass
class FakeModelOutput:
    recon_from_robot: torch.Tensor
    recon_from_human: torch.Tensor


class FakeModel:
    def __call__(self, robot_window: torch.Tensor, human_window: torch.Tensor) -> FakeModelOutput:
        robot_base = robot_window if robot_window.numel() else torch.zeros(
            (human_window.shape[0], 9),
            dtype=human_window.dtype,
            device=human_window.device,
        )
        return FakeModelOutput(
            recon_from_robot=robot_base + 10.0,
            recon_from_human=torch.full(
                (human_window.shape[0], 9),
                7.0,
                dtype=human_window.dtype,
                device=human_window.device,
            ),
        )


class ProjectHumanModel:
    def __call__(self, robot_window: torch.Tensor, human_window: torch.Tensor) -> FakeModelOutput:
        batch = human_window.shape[0]
        projected = human_window[:, :9]
        robot_base = robot_window if robot_window.numel() else torch.zeros(
            (batch, projected.shape[-1]),
            dtype=human_window.dtype,
            device=human_window.device,
        )
        return FakeModelOutput(
            recon_from_robot=robot_base,
            recon_from_human=projected,
        )


def _make_source_bundle() -> InferenceSourceBundle:
    return InferenceSourceBundle(
        fps=20,
        center_indices=torch.tensor([1, 2], dtype=torch.long),
        window_offsets=torch.tensor([-1, 0, 1], dtype=torch.long),
        robot_features=None,
        human_features=torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6),
        robot_anchor_pos_w=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        human_body_pos_w=torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
                [[3.0, 0.0, 0.0], [3.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        robot_joint_names=["joint_a", "joint_b", "joint_c"],
        robot_body_names=["base", "torso_link"],
        human_body_names=["Hips", "Spine1"],
        robot_anchor_body="torso_link",
        human_anchor_body="Hips",
        display_human_body_names=["Hips", "Spine1"],
    )


def _make_human_only_npz_from_raw_sample(raw_path: Path, output_path: Path) -> Path:
    with np.load(raw_path, allow_pickle=True) as data:
        payload = {
            "fps": np.asarray(data["fps"]).copy(),
            "scalar_first": np.asarray(False),
            "human_joint_names": np.asarray(data["human_joint_names"], dtype=object),
            "human_parent_indices": np.asarray(data["human_parent_indices"], dtype=np.int32),
            "human_reference_local_transforms": np.asarray(data["human_reference_local_transforms"], dtype=np.float32),
            "human_local_transforms": np.asarray(data["human_local_transforms"], dtype=np.float32),
            "human_global_pos": np.asarray(data["human_global_pos"], dtype=np.float32),
            "human_global_quat": np.asarray(data["human_global_quat"], dtype=np.float32),
            "timeline_frame_indices": np.arange(
                np.asarray(data["human_local_transforms"]).shape[0],
                dtype=np.int32,
            ),
        }
    np.savez(output_path, **payload)
    return output_path


def test_reconstruct_from_source_human_path_supports_human_only_bundle() -> None:
    from motion_reconstruction.evaluation.reconstruct import reconstruct_from_source_bundle

    result = reconstruct_from_source_bundle(
        source=_make_source_bundle(),
        model=FakeModel(),
        normalizers={"robot": IdentityNormalizer(), "human": IdentityNormalizer()},
        history_index=1,
        robot_dim=3,
        inference_path="human",
        batch_size=16,
    )

    assert result.original_robot_feature is None
    assert result.recon_from_robot_feature is None
    assert result.recon_from_human_feature.shape == (2, 3)
    np.testing.assert_allclose(result.recon_from_human_feature, 7.0)
    np.testing.assert_allclose(
        result.robot_anchor_pos_w,
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )


def test_reconstruct_from_source_robot_path_requires_robot_features() -> None:
    from motion_reconstruction.evaluation.reconstruct import reconstruct_from_source_bundle

    with pytest.raises(ValueError, match="robot_features"):
        reconstruct_from_source_bundle(
            source=_make_source_bundle(),
            model=FakeModel(),
            normalizers={"robot": IdentityNormalizer(), "human": IdentityNormalizer()},
            history_index=1,
            robot_dim=3,
            inference_path="robot",
            batch_size=4,
        )


def test_reconstruction_metrics_skip_missing_branches() -> None:
    result = ReconstructionResult(
        fps=20,
        center_indices=np.array([1, 2], dtype=np.int64),
        original_robot_feature=None,
        recon_from_robot_feature=None,
        recon_from_human_feature=np.ones((2, 3), dtype=np.float32),
        robot_anchor_pos_w=np.zeros((2, 3), dtype=np.float32),
        human_body_pos_w=np.zeros((2, 2, 3), dtype=np.float32),
        robot_joint_names=[],
        robot_body_names=[],
        human_body_names=["Hips", "Spine1"],
        robot_anchor_body="torso_link",
        human_anchor_body="Hips",
        display_human_body_names=["Hips", "Spine1"],
    )

    metrics = result.metrics()

    assert metrics == {}


def test_visualize_cli_parser_supports_source_and_inference_path() -> None:
    from motion_reconstruction.cli.visualize import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "latest.pt",
            "--xml-path",
            "robot.xml",
            "--source",
            "hdf5-human",
            "--motion-npz",
            "annotation_soma.npz",
            "--inference-path",
            "human",
        ]
    )

    assert args.source == "hdf5-human"
    assert args.motion_npz == Path("annotation_soma.npz")
    assert args.inference_path == "human"


def test_validate_visualization_pair_rejects_robot_only_views_for_human_only_result() -> None:
    from motion_reconstruction.visualization.mujoco_viewer import validate_reconstruction_for_pair

    result = ReconstructionResult(
        fps=20,
        center_indices=np.array([1, 2], dtype=np.int64),
        original_robot_feature=None,
        recon_from_robot_feature=None,
        recon_from_human_feature=np.ones((2, 3), dtype=np.float32),
        robot_anchor_pos_w=np.zeros((2, 3), dtype=np.float32),
        human_body_pos_w=np.zeros((2, 2, 3), dtype=np.float32),
        robot_joint_names=[],
        robot_body_names=[],
        human_body_names=["Hips", "Spine1"],
        robot_anchor_body="torso_link",
        human_anchor_body="Hips",
        display_human_body_names=["Hips", "Spine1"],
    )

    with pytest.raises(ValueError, match="pair=robot"):
        validate_reconstruction_for_pair(result=result, pair="robot")
    with pytest.raises(ValueError, match="pair=both"):
        validate_reconstruction_for_pair(result=result, pair="both")
    validate_reconstruction_for_pair(result=result, pair="human")


def test_hdf5_parse_wrapper_parser_exposes_motion_npz_and_checkpoint() -> None:
    import importlib.util
    import sys

    wrapper_path = REPO_ROOT / "hdf5_parse" / "visualize_hdf5_soma_npz.py"
    spec = importlib.util.spec_from_file_location("visualize_hdf5_soma_npz", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.build_arg_parser().parse_args(
        [
            "--config",
            "cfg.yaml",
            "--checkpoint",
            "latest.pt",
            "--xml-path",
            "robot.xml",
        ]
    )

    assert args.motion_npz == Path("hdf5_parse/out/annotation_soma.npz")


def test_visualize_hdf5_human_npz_forwards_shared_package_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from motion_reconstruction.visualization.api import visualize_hdf5_human_npz
    from motion_reconstruction.config.schema import MotionReconstructionConfig

    calls: dict[str, object] = {}

    def fake_reconstruct_motion(**kwargs):
        calls["reconstruct"] = kwargs
        return ReconstructionResult(
            fps=20,
            center_indices=np.array([1], dtype=np.int64),
            original_robot_feature=None,
            recon_from_robot_feature=None,
            recon_from_human_feature=np.ones((1, 3), dtype=np.float32),
            robot_anchor_pos_w=np.zeros((1, 3), dtype=np.float32),
            human_body_pos_w=np.zeros((1, 2, 3), dtype=np.float32),
            robot_joint_names=[],
            robot_body_names=[],
            human_body_names=["Hips", "Spine1"],
            robot_anchor_body="torso_link",
            human_anchor_body="Hips",
            display_human_body_names=["Hips", "Spine1"],
        )

    def fake_play_reconstruction(**kwargs):
        calls["play"] = kwargs

    monkeypatch.setattr("motion_reconstruction.visualization.api.reconstruct_motion", fake_reconstruct_motion)
    monkeypatch.setattr("motion_reconstruction.visualization.api.play_reconstruction", fake_play_reconstruction)

    config = MotionReconstructionConfig()
    visualize_hdf5_human_npz(
        config=config,
        checkpoint_path="latest.pt",
        xml_path="robot.xml",
        motion_npz="annotation_soma.npz",
        max_frames=4,
    )

    assert calls["reconstruct"]["source"] == "hdf5-human"
    assert calls["reconstruct"]["inference_path"] == "human"
    assert calls["reconstruct"]["motion_npz"] == "annotation_soma.npz"
    assert calls["play"]["pair"] == "human"


def test_reconstruct_from_hdf5_human_matches_raw_human_path_for_stripped_raw_npz(tmp_path: Path) -> None:
    from motion_reconstruction.evaluation.reconstruct import reconstruct_from_source_bundle
    from motion_reconstruction.inference.sources import build_hdf5_human_source, build_raw_source
    from motion_reconstruction.pipeline import ResolvedMotionFiles

    raw_sample = (
        REPO_ROOT
        / "soma-retargeter"
        / "assets"
        / "motions"
        / "soma_uniform_bvh_export"
        / "240918"
        / "body_check_001__A548.npz"
    )
    assert raw_sample.is_file()
    human_only_sample = _make_human_only_npz_from_raw_sample(raw_sample, tmp_path / "human_only_from_raw.npz")

    config = MotionReconstructionConfig()
    config.train.history = 0
    config.train.future = 0

    raw_bundle = build_raw_source(
        config=config,
        device="cpu",
        resolved=ResolvedMotionFiles(paths=[raw_sample], groups=["test"]),
        progress=False,
    )
    hdf_bundle = build_hdf5_human_source(
        motion_npz=human_only_sample,
        config=config,
        feature_schema={
            "robot_joint_names": raw_bundle.robot_joint_names,
            "robot_body_names": raw_bundle.robot_body_names,
        },
        device="cpu",
    )

    np.testing.assert_allclose(
        hdf_bundle.human_features.cpu().numpy(),
        raw_bundle.human_features.cpu().numpy(),
        atol=1e-6,
    )

    result_raw = reconstruct_from_source_bundle(
        source=raw_bundle,
        model=ProjectHumanModel(),
        normalizers={"robot": IdentityNormalizer(), "human": IdentityNormalizer()},
        history_index=0,
        robot_dim=9,
        inference_path="human",
        batch_size=512,
    )
    result_hdf = reconstruct_from_source_bundle(
        source=hdf_bundle,
        model=ProjectHumanModel(),
        normalizers={"robot": IdentityNormalizer(), "human": IdentityNormalizer()},
        history_index=0,
        robot_dim=9,
        inference_path="human",
        batch_size=512,
    )

    np.testing.assert_array_equal(result_hdf.center_indices, result_raw.center_indices)
    np.testing.assert_allclose(result_hdf.human_body_pos_w, result_raw.human_body_pos_w, atol=1e-6)
    np.testing.assert_allclose(result_hdf.recon_from_human_feature, result_raw.recon_from_human_feature, atol=1e-6)


def test_build_inference_config_prefers_checkpoint_model_and_window_settings() -> None:
    from motion_reconstruction.evaluation.reconstruct import build_inference_config
    from motion_reconstruction.config.schema import MotionReconstructionConfig

    base = MotionReconstructionConfig()
    base.data.files = ["new_motion.npz"]
    base.train.history = 0
    base.train.future = 9
    base.model.latent_dim = 16
    payload = {
        "config": {
            "features": {
                "robot_anchor_body": "torso_link",
                "human_anchor_body": "Hips",
                "human_body_names": ["Spine1", "Head"],
            },
            "model": {
                "latent_dim": 64,
                "robot_encoder_hidden_dims": [1024, 512, 256],
                "human_encoder_hidden_dims": [1024, 512, 256],
                "decoder_hidden_dims": [256, 512, 1024],
                "activation": "elu",
                "quantizer": {
                    "type": "ifsq",
                    "levels": 17,
                    "do_simple_bound": True,
                    "act_fun": "scale_sigmoid_16",
                    "eps": 0.001,
                },
            },
            "train": {
                "history": 5,
                "future": 5,
            },
        }
    }

    merged = build_inference_config(base, payload)

    assert merged.data.files == ["new_motion.npz"]
    assert merged.train.history == 5
    assert merged.train.future == 5
    assert merged.model.latent_dim == 64
    assert merged.features.human_body_names == ["Spine1", "Head"]
