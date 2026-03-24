"""Observation configuration used by the simple observation manager."""

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

ObsCfg = G1ObsCfg