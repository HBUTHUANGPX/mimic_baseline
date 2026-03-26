"""Observation configuration used by the simple observation manager."""

import sys

from awesome_deploy.utils.cfg import resolve_robot_name
from awesome_deploy.utils.observation_manager import (
    TermCfg,
    GroupCfg,
)


class G1ObsCfg:
    """Top-level observation configuration container.

    Each attribute on this class is interpreted by ``SimpleObservationManager``
    as one observation group. The nested group classes describe both the order
    of observation terms and any per-term overrides.
    """

    class PolicyCfg(GroupCfg):
        """Observation terms consumed by the deployed policy model."""

        motion_joint_pos_command = TermCfg()
        motion_joint_vel_command = TermCfg()
        motion_ref_ori_b = TermCfg()
        base_ang_vel = TermCfg()
        joint_pos = TermCfg()
        joint_vel = TermCfg()
        actions = TermCfg()

    policy = PolicyCfg()
    input_group_map = {
        "policy_obs": "policy",
    }

class Q1ObsCfg:
    """Top-level observation configuration container.

    Each attribute on this class is interpreted by ``SimpleObservationManager``
    as one observation group. The nested group classes describe both the order
    of observation terms and any per-term overrides.
    """

    class PolicyCfg(GroupCfg):
        """Observation terms consumed by the deployed policy model."""

        joint_pos_delta = TermCfg()
        robot_joint_pos = TermCfg()
        motion_ref_ori_b = TermCfg()
        base_ang_vel = TermCfg(history_length=8)
        joint_pos = TermCfg(history_length=8)
        joint_vel = TermCfg(history_length=8)
        actions = TermCfg()
    
    class PolicyWindowCfg(GroupCfg):
        """Observation terms consumed by the deployed policy model."""

        joint_pos_delta_window = TermCfg()
        robot_joint_pos_window = TermCfg()
        motion_ref_ori_b_window = TermCfg()

    policy = PolicyCfg()
    policy_window = PolicyWindowCfg()
    input_group_map = {
        "actor_obs": "policy",
        "actor_fsq_obs": "policy_window",
        "policy_obs": "policy",
    }


OBS_CFG_REGISTRY = {
    "g1": G1ObsCfg,
    "q1": Q1ObsCfg,
}


def resolve_obs_name(
    argv: list[str] | None = None,
    default: str | None = None,
) -> str:
    """Resolves the active observation configuration name.

    Args:
        argv: Optional argument list. Defaults to ``sys.argv``.
        default: Optional fallback. When omitted, the active ``robot_name`` is
            used as the default observation name.

    Returns:
        Resolved observation configuration identifier.
    """
    argv = argv if argv is not None else sys.argv
    for index, arg in enumerate(argv):
        if arg.startswith("obs_name="):
            return arg.split("=", 1)[1]
        if arg.startswith("--obs_name="):
            return arg.split("=", 1)[1]
        if arg == "--obs_name" and index + 1 < len(argv):
            return argv[index + 1]
    return default or resolve_robot_name(argv=argv)


def build_obs_cfg(obs_name: str):
    """Builds one registered observation configuration instance."""
    try:
        obs_cfg_cls = OBS_CFG_REGISTRY[obs_name]
    except KeyError as exc:
        supported = ", ".join(sorted(OBS_CFG_REGISTRY))
        raise ValueError(
            f"Unknown obs_name '{obs_name}'. Supported observation configs: {supported}"
        ) from exc
    return obs_cfg_cls()


obs_cfg = build_obs_cfg(resolve_obs_name())


def get_obs_cfg(obs_name=None):
    """Returns the active observation config or resolves one on demand."""
    resolved_obs_name = obs_name or resolve_obs_name()
    return build_obs_cfg(resolved_obs_name)
