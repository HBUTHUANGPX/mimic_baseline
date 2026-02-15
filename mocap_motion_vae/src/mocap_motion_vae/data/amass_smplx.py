"""AMASS SMPL-X 数据解析与 MotionBank 构建工具。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from smplx.joint_names import JOINT_NAMES
import smplx

from .bank import ClipData, MotionBank

@dataclass
class SMPLXFieldSpec:
    """指定训练时暴露哪些 SMPL-X 字段（帧级/静态）。

    Attributes:
        frame_fields: 帧级字段名列表。
            可包含 "pose_body"、"root_orient"、"trans"、"joints"、"vertices"、"full_pose" 等。
        static_fields: 静态字段名列表。
    """

    frame_fields: Tuple[str, ...] = ("pose_body", "root_orient", "trans", "joints")
    static_fields: Tuple[str, ...] = ("betas",)


@dataclass
class SMPLXClip:
    """单个 SMPL-X 片段的原始数据容器。

    Attributes:
        path: 文件路径。
        fps: 帧率（Hz）。
        pose_body: 身体姿态，形状 (T, 63)。
        root_orient: 根关节旋转，形状 (T, 3)。
        trans: 平移，形状 (T, 3)。
        betas: 形体参数，形状 (B,) 或 (1, B)。
        gender: 性别字符串。
        meta: 其他元信息。
    """

    path: str
    fps: float
    pose_body: np.ndarray
    root_orient: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    gender: str
    meta: Dict[str, object]

    def to_clip_data(self, spec: SMPLXFieldSpec, device: str | torch.device) -> ClipData:
        """将 SMPLXClip 转为通用 ClipData，用于构建 MotionBank。

        Args:
            spec: 字段选择配置。
            device: 张量放置设备。

        Returns:
            ClipData 实例。
        """
        frames: Dict[str, torch.Tensor] = {}
        for name in spec.frame_fields:
            frames[name] = torch.tensor(self._get_frame_field(name), dtype=torch.float32, device=device)

        static: Dict[str, torch.Tensor] = {}
        for name in spec.static_fields:
            static[name] = torch.tensor(self._get_static_field(name), dtype=torch.float32, device=device)

        return ClipData(
            name=Path(self.path).stem,
            fps=float(self.fps),
            frames=frames,
            static=static,
            meta=dict(self.meta),
        )

    def _get_frame_field(self, name: str) -> np.ndarray:
        """读取帧级字段（T, D）。

        Args:
            name: 字段名。

        Returns:
            对应字段的 NumPy 数组。

        Raises:
            KeyError: 字段不存在。
        """
        if name == "pose_body":
            return self.pose_body
        if name == "root_orient":
            return self.root_orient
        if name == "trans":
            return self.trans
        if name in ("joints", "vertices", "full_pose") and name not in self.meta:
            raise KeyError(
                f"Field '{name}' requires SMPL-X model outputs. "
                "Please provide SMPLX_MODEL_PATH or smplx_model_path."
            )
        if name in self.meta:
            value = self.meta[name]
            if isinstance(value, np.ndarray):
                return value
        raise KeyError(f"Unknown frame field '{name}'.")

    def _get_static_field(self, name: str) -> np.ndarray:
        """读取静态字段（与时间无关）。

        Args:
            name: 字段名。

        Returns:
            对应字段的 NumPy 数组。

        Raises:
            KeyError: 字段不存在。
        """
        if name == "betas":
            betas = np.asarray(self.betas)
            return betas.reshape(-1)
        if name == "gender_id":
            # 将性别映射为数值，便于拼接到模型输入
            mapping = {"male": 0.0, "female": 1.0, "neutral": 2.0}
            return np.asarray([mapping.get(self.gender.lower(), -1.0)], dtype=np.float32)
        if name == "height":
            # 用 betas[0] 估计身高（与你提供的示例保持一致）
            betas = np.asarray(self.betas).reshape(-1)
            height = 1.66 + 0.1 * float(betas[0]) if betas.size > 0 else 1.66
            return np.asarray([height], dtype=np.float32)
        if name in self.meta:
            value = self.meta[name]
            if isinstance(value, np.ndarray):
                return value
        raise KeyError(f"Unknown static field '{name}'.")


class SMPLXClipParser:
    """解析 AMASS SMPL-X npz 文件并生成 SMPLXClip。

    默认会使用 smplx 库加载模型，并输出 joints/vertices/full_pose 等结果。
    需要提供模型路径（或设置环境变量 SMPLX_MODEL_PATH）。
    """

    def __init__(
        self,
        smplx_model_path: Optional[str] = None,
        model_type: str = "smplx",
        use_pca: bool = False,
        num_betas: Optional[int] = None,
        device: str | torch.device = "cpu",
        allow_pickle: bool = True,
    ) -> None:
        """创建解析器。

        Args:
            smplx_model_path: SMPL-X 模型目录路径。
                若为空，将尝试读取环境变量 `SMPLX_MODEL_PATH`。
            model_type: 模型类型，默认 "smplx"。
            use_pca: 是否启用 PCA 手部姿态。
            num_betas: betas 维度（可选）。
            device: 张量与模型放置设备。
            allow_pickle: 是否允许 pickle 读取（AMASS 常用）。
        """
        self.allow_pickle = allow_pickle
        self.model_path = smplx_model_path or os.getenv("SMPLX_MODEL_PATH")
        self.model_type = model_type
        self.use_pca = use_pca
        self.num_betas = num_betas
        self.device = torch.device(device)
        self._models: Dict[str, torch.nn.Module] = {}

    def parse(self, path: str) -> SMPLXClip:
        """读取单个 npz 文件，返回结构化的 SMPLXClip。

        Args:
            path: npz 文件路径。

        Returns:
            SMPLXClip 实例。
        """
        data = np.load(path, allow_pickle=self.allow_pickle)

        fps = self._get_fps(data)
        gender = self._get_gender(data)
        betas = self._get_betas(data)
        num_betas = int(self.num_betas or betas.reshape(-1).shape[0])
        model = self._get_body_model(gender, num_betas=num_betas)

        smplx_inputs = self._extract_smplx_inputs(data, model, betas=betas)
        pose_body = smplx_inputs["body_pose"]
        root_orient = smplx_inputs["global_orient"]
        trans = smplx_inputs["transl"]
        betas = smplx_inputs["betas"]

        meta: Dict[str, object] = {"path": path}
        smplx_meta = self._run_smplx_model(
            model=model,
            body_pose=pose_body,
            global_orient=root_orient,
            transl=trans,
            betas=betas,
            extra_inputs=smplx_inputs,
        )
        meta.update(smplx_meta)

        return SMPLXClip(
            path=path,
            fps=fps,
            pose_body=pose_body,
            root_orient=root_orient,
            trans=trans,
            betas=betas,
            gender=gender,
            meta=meta,
        )

    def _get_fps(self, data: np.lib.npyio.NpzFile) -> float:
        """兼容不同字段名的 fps 读取。

        Args:
            data: npz 数据对象。

        Returns:
            帧率（Hz）。

        Raises:
            KeyError: 未找到 fps 字段。
        """
        for key in ("fps", "mocap_framerate", "mocap_frame_rate", "frame_rate"):
            if key in data:
                return float(np.asarray(data[key]).item())
        raise KeyError(
            "Could not find fps in npz (expected 'fps', 'mocap_framerate', or 'mocap_frame_rate')."
        )

    def _get_trans(self, data: np.lib.npyio.NpzFile) -> np.ndarray:
        """位移字段兼容 trans/transl。

        Args:
            data: npz 数据对象。

        Returns:
            trans 数组。

        Raises:
            KeyError: 未找到字段。
        """
        for key in ("trans", "transl"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32)
        raise KeyError("Could not find trans in npz (expected 'trans' or 'transl').")

    def _get_betas(self, data: np.lib.npyio.NpzFile) -> np.ndarray:
        """SMPL-X 体型参数。

        Args:
            data: npz 数据对象。

        Returns:
            betas 数组。

        Raises:
            KeyError: 未找到字段。
        """
        if "betas" in data:
            return np.asarray(data["betas"], dtype=np.float32)
        raise KeyError("Could not find betas in npz (expected 'betas').")

    def _get_gender(self, data: np.lib.npyio.NpzFile) -> str:
        """读取性别字段，兼容 bytes/ndarray。

        Args:
            data: npz 数据对象。

        Returns:
            性别字符串。
        """
        if "gender" not in data:
            return "neutral"
        gender = data["gender"]
        if isinstance(gender, np.ndarray):
            if gender.shape == ():
                gender = gender.item()
            elif gender.size == 1:
                gender = gender.reshape(-1)[0]
        if isinstance(gender, bytes):
            return gender.decode("utf-8")
        return str(gender)

    def _get_body_model(self, gender: str, num_betas: int) -> torch.nn.Module:
        """按性别创建并缓存 SMPL-X 模型。

        Args:
            gender: 性别字符串。
            num_betas: betas 维度。

        Returns:
            SMPL-X 模型实例。

        Raises:
            RuntimeError: 未提供模型路径。
        """
        gender_key = gender.lower()
        cache_key = f"{gender_key}:{num_betas}"
        if cache_key in self._models:
            return self._models[cache_key]
        if self.model_path is None:
            raise RuntimeError(
                "SMPL-X 模型路径未设置。请传入 smplx_model_path 或设置环境变量 SMPLX_MODEL_PATH。"
            )
        create_kwargs = {
            "model_type": self.model_type,
            "gender": gender_key,
            "use_pca": self.use_pca,
        }
        create_kwargs["num_betas"] = int(num_betas)
        model = smplx.create(self.model_path, **create_kwargs).to(self.device)
        self._models[cache_key] = model
        return model

    def _run_smplx_model(
        self,
        model: torch.nn.Module,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray,
        extra_inputs: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """使用 smplx 模型生成 joints/vertices/full_pose 等字段。

        Args:
            model: SMPL-X 模型实例。
            body_pose: 身体姿态。
            global_orient: 根关节旋转。
            transl: 平移。
            betas: 形体参数。
            extra_inputs: 其他输入（手部/面部/表情等）。

        Returns:
            包含 joints/vertices/full_pose 等字段的字典。
        """
        num_frames = body_pose.shape[0]

        betas_array = np.asarray(betas, dtype=np.float32).reshape(-1)
        num_betas_model = int(getattr(model, "num_betas", betas_array.shape[0]))
        if betas_array.shape[0] < num_betas_model:
            pad = np.zeros((num_betas_model - betas_array.shape[0],), dtype=np.float32)
            betas_array = np.concatenate([betas_array, pad], axis=0)
        elif betas_array.shape[0] > num_betas_model:
            betas_array = betas_array[:num_betas_model]
        betas_tensor = torch.tensor(betas_array, dtype=torch.float32, device=self.device).view(1, -1)
        betas_tensor = betas_tensor.expand(num_frames, -1)

        body_pose = torch.tensor(body_pose, dtype=torch.float32, device=self.device)
        global_orient = torch.tensor(global_orient, dtype=torch.float32, device=self.device)
        transl = torch.tensor(transl, dtype=torch.float32, device=self.device)

        left_hand_pose = torch.tensor(
            extra_inputs["left_hand_pose"], dtype=torch.float32, device=self.device
        )
        right_hand_pose = torch.tensor(
            extra_inputs["right_hand_pose"], dtype=torch.float32, device=self.device
        )
        jaw_pose = torch.tensor(extra_inputs["jaw_pose"], dtype=torch.float32, device=self.device)
        leye_pose = torch.tensor(extra_inputs["leye_pose"], dtype=torch.float32, device=self.device)
        reye_pose = torch.tensor(extra_inputs["reye_pose"], dtype=torch.float32, device=self.device)

        expression = torch.tensor(
            extra_inputs["expression"], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            smplx_output = model(
                betas=betas_tensor,
                global_orient=global_orient,
                body_pose=body_pose,
                transl=transl,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                expression=expression,
                return_full_pose=True,
            )

        meta: Dict[str, np.ndarray] = {}
        if hasattr(smplx_output, "joints") and smplx_output.joints is not None:
            meta["joints"] = smplx_output.joints.detach().cpu().numpy()
            meta["joint_names"] = np.array(JOINT_NAMES)
        if hasattr(smplx_output, "vertices") and smplx_output.vertices is not None:
            meta["vertices"] = smplx_output.vertices.detach().cpu().numpy()
        if hasattr(smplx_output, "full_pose") and smplx_output.full_pose is not None:
            meta["full_pose"] = smplx_output.full_pose.detach().cpu().numpy()
        return meta

    def _extract_smplx_inputs(
        self, data: np.lib.npyio.NpzFile, model: torch.nn.Module, betas: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """使用 SMPL-X 模型定义的维度解析输入字段。

        Args:
            data: npz 数据对象。
            model: SMPL-X 模型实例。
            betas: 形体参数数组。

        Returns:
            可直接喂给 SMPL-X 的输入字典。

        Raises:
            KeyError: 缺失关键字段。
            ValueError: 姿态维度不匹配。
        """
        transl = self._get_trans(data)

        poses = None
        if "poses" in data:
            poses = np.asarray(data["poses"], dtype=np.float32)

        if poses is None:
            global_orient = self._get_required_array(data, ("root_orient", "global_orient"))
            body_pose = self._get_required_array(data, ("pose_body", "body_pose"))
            poses = self._assemble_full_pose(
                global_orient=global_orient,
                body_pose=body_pose,
                model=model,
                data=data,
            )

        global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose = (
            self._split_full_pose(poses, model)
        )

        num_frames = poses.shape[0]
        expression = self._get_optional_array(
            data, "expression", default_shape=(num_frames, model.num_expression_coeffs)
        )

        return {
            "betas": betas,
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "jaw_pose": jaw_pose,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "expression": expression,
        }

    def _get_required_array(
        self, data: np.lib.npyio.NpzFile, keys: Tuple[str, ...]
    ) -> np.ndarray:
        """从候选键中读取必需数组。

        Args:
            data: npz 数据对象。
            keys: 候选字段名。

        Returns:
            读取到的 NumPy 数组。

        Raises:
            KeyError: 未找到任意候选字段。
        """
        for key in keys:
            if key in data:
                return np.asarray(data[key], dtype=np.float32)
        raise KeyError(f"Could not find required fields: {keys}")

    def _get_optional_array(
        self,
        data: np.lib.npyio.NpzFile,
        key: str,
        default_shape: Tuple[int, int],
    ) -> np.ndarray:
        """读取可选字段，不存在则返回零数组。

        Args:
            data: npz 数据对象。
            key: 字段名。
            default_shape: 默认形状。

        Returns:
            NumPy 数组。
        """
        if key in data:
            return np.asarray(data[key], dtype=np.float32)
        return np.zeros(default_shape, dtype=np.float32)

    def _split_full_pose(
        self, poses: np.ndarray, model: torch.nn.Module
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """按 SMPL-X 模型定义切分 full pose。

        Args:
            poses: full pose 数组，形状 (T, D)。
            model: SMPL-X 模型实例。

        Returns:
            (global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose)

        Raises:
            ValueError: 维度不匹配。
        """
        poses = np.asarray(poses, dtype=np.float32)
        if poses.ndim != 2:
            raise ValueError(f"Expected poses shape (T, D), got {poses.shape}")

        num_body = int(model.NUM_BODY_JOINTS)
        num_hand = int(model.NUM_HAND_JOINTS)
        num_face = int(getattr(model, "NUM_FACE_JOINTS", 3))

        body_only_dim = 3 + num_body * 3
        expected = body_only_dim + num_face * 3 + 2 * num_hand * 3
        if poses.shape[1] < expected:
            if poses.shape[1] == body_only_dim:
                num_frames = poses.shape[0]
                zero_face = np.zeros((num_frames, num_face * 3), dtype=np.float32)
                zero_hand = np.zeros((num_frames, num_hand * 3), dtype=np.float32)
                poses = np.concatenate([poses, zero_face, zero_hand, zero_hand], axis=1)
            else:
                raise ValueError(
                    f"Full pose dim too small. Expected {body_only_dim} or >= {expected}, "
                    f"got {poses.shape[1]}."
                )

        cursor = 0
        global_orient = poses[:, cursor : cursor + 3]
        cursor += 3
        body_pose = poses[:, cursor : cursor + num_body * 3]
        cursor += num_body * 3
        jaw_pose = poses[:, cursor : cursor + 3]
        cursor += 3
        leye_pose = poses[:, cursor : cursor + 3]
        cursor += 3
        reye_pose = poses[:, cursor : cursor + 3]
        cursor += 3
        left_hand_pose = poses[:, cursor : cursor + num_hand * 3]
        cursor += num_hand * 3
        right_hand_pose = poses[:, cursor : cursor + num_hand * 3]
        return (
            global_orient,
            body_pose,
            jaw_pose,
            leye_pose,
            reye_pose,
            left_hand_pose,
            right_hand_pose,
        )

    def _assemble_full_pose(
        self,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        model: torch.nn.Module,
        data: np.lib.npyio.NpzFile,
    ) -> np.ndarray:
        """使用 SMPL-X 模型维度组装 full pose。

        Args:
            global_orient: 根关节旋转。
            body_pose: 身体姿态。
            model: SMPL-X 模型实例。
            data: npz 数据对象（可选字段读取）。

        Returns:
            full pose 数组。
        """
        num_frames = body_pose.shape[0]
        num_hand = int(model.NUM_HAND_JOINTS)

        jaw_pose = self._get_optional_array(data, "jaw_pose", (num_frames, 3))
        leye_pose = self._get_optional_array(data, "leye_pose", (num_frames, 3))
        reye_pose = self._get_optional_array(data, "reye_pose", (num_frames, 3))
        left_hand_pose = self._get_optional_array(data, "left_hand_pose", (num_frames, num_hand * 3))
        right_hand_pose = self._get_optional_array(data, "right_hand_pose", (num_frames, num_hand * 3))

        return np.concatenate(
            [
                global_orient,
                body_pose,
                jaw_pose,
                leye_pose,
                reye_pose,
                left_hand_pose,
                right_hand_pose,
            ],
            axis=1,
        )


def discover_amass_smplx_files(
    root: str | Path,
    pattern: str = "**/*.npz",
    limit: Optional[int] = None,
) -> List[str]:
    """递归扫描 AMASS/SMPL-X npz 文件。

    Args:
        root: 数据根目录。
        pattern: 递归匹配模式。
        limit: 最多返回的文件数量。

    Returns:
        文件路径列表（排序后）。
    """
    base = Path(root)
    files = [str(p) for p in base.rglob(pattern) if p.is_file()]
    files.sort()
    if limit is not None:
        files = files[: int(limit)]
    return files


def build_amass_smplx_bank(
    files: Sequence[str],
    spec: SMPLXFieldSpec,
    device: str | torch.device = "cpu",
    parser: Optional[SMPLXClipParser] = None,
    smplx_model_path: Optional[str] = None,
    strict: bool = True,
) -> MotionBank:
    """解析文件列表并构建 MotionBank。

    Args:
        files: npz 文件路径列表。
        spec: 字段选择配置。
        device: 张量放置设备。
        parser: 解析器实例（可选）。
        smplx_model_path: SMPL-X 模型目录路径（可选）。
        strict: 解析失败是否抛异常。

    Returns:
        MotionBank 实例。
    """
    if parser is None:
        parser = SMPLXClipParser(
            smplx_model_path=smplx_model_path,
            device=device,
        )

    clips: List[ClipData] = []
    for path in files:
        try:
            clip = parser.parse(path)
            clips.append(clip.to_clip_data(spec, device))
        except Exception:
            if strict:
                raise
            continue

    return MotionBank.from_clips(clips, device=device)
