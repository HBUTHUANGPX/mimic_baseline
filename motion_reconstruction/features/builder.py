"""从 raw motion tensor 构建可复用的 robot/human feature。

FeatureBuilder 是 motion 语义到网络输入的唯一转换层。raw loader 只保证
字段和顺序，模型只接收 feature tensor，因此该模块是多工程兼容的关键边界。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from motion_reconstruction.data.raw_motion import RawMotionDataset
from motion_reconstruction.features.rotation import quat_inverse_rotate_wxyz, quat_to_rot6d_wxyz


DEFAULT_HUMAN_BODY_NAMES = [
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
]


@dataclass
class FeatureBuilderConfig:
    """feature 选择配置。

    robot 使用全部 joint_pos；human 只使用这里列出的 body 在 anchor frame
    下的位置。
    """

    robot_anchor_body: str = "torso_link"
    human_anchor_body: str = "Hips"
    human_body_names: list[str] = field(default_factory=lambda: list(DEFAULT_HUMAN_BODY_NAMES))


@dataclass
class FeatureSchema:
    """可序列化的 feature 构建描述。

    checkpoint 会保存该 schema，方便之后还原网络输入含义。
    """

    robot_anchor_body: str
    human_anchor_body: str
    human_body_names: list[str]
    robot_joint_names: list[str]
    robot_body_names: list[str]
    source_human_body_names: list[str]
    robot_feature_dim: int
    human_feature_dim: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureBundle:
    """逐帧 robot/human feature tensor 及其 schema。"""

    robot: torch.Tensor
    human: torch.Tensor
    schema: FeatureSchema


class FeatureBuilder:
    """将 raw motion tensor 转成逐帧网络 feature。

    中文输出：
    - robot: `[T, 6 + num_robot_joints]`
    - human: `[T, 6 + 3 * num_selected_human_bodies]`
    """

    def __init__(self, config: FeatureBuilderConfig):
        self.config = config

    def build(self, raw: RawMotionDataset) -> FeatureBundle:
        robot_anchor_idx = _index(raw.robot_body_names, self.config.robot_anchor_body, "robot body")
        human_anchor_idx = _index(raw.human_body_names, self.config.human_anchor_body, "human body")
        human_body_indices = [
            _index(raw.human_body_names, name, "human body") for name in self.config.human_body_names
        ]

        robot_anchor_rot6d = quat_to_rot6d_wxyz(raw.body_quat_w[:, robot_anchor_idx])
        robot = torch.cat((robot_anchor_rot6d, raw.joint_pos), dim=-1)

        # 中文：human 位置目标是“相对 anchor body 的局部 frame 位移”，这样减少
        # world 平移对重构任务的干扰。
        human_anchor_quat = raw.human_body_quat_w[:, human_anchor_idx]
        human_anchor_rot6d = quat_to_rot6d_wxyz(human_anchor_quat)
        anchor_pos = raw.human_body_pos_w[:, human_anchor_idx]
        selected_pos = raw.human_body_pos_w[:, human_body_indices]
        rel_world = selected_pos - anchor_pos[:, None, :]
        expanded_anchor_quat = human_anchor_quat[:, None, :].expand(-1, len(human_body_indices), -1)
        rel_anchor = quat_inverse_rotate_wxyz(expanded_anchor_quat, rel_world).reshape(raw.num_frames, -1)
        human = torch.cat((human_anchor_rot6d, rel_anchor), dim=-1)

        schema = FeatureSchema(
            robot_anchor_body=self.config.robot_anchor_body,
            human_anchor_body=self.config.human_anchor_body,
            human_body_names=list(self.config.human_body_names),
            robot_joint_names=list(raw.robot_joint_names),
            robot_body_names=list(raw.robot_body_names),
            source_human_body_names=list(raw.human_body_names),
            robot_feature_dim=int(robot.shape[-1]),
            human_feature_dim=int(human.shape[-1]),
        )
        return FeatureBundle(robot=robot, human=human, schema=schema)


def _index(names: list[str], name: str, label: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown {label} '{name}'. Available names: {names}") from exc
