from awesome_deploy.utils.observation_manager import (
    SimpleObservationManager,
    TermCfg,
    GroupCfg,
)


class ObsCfg:
    """观测总配置：每个属性是一个 GroupCfg 实例。"""

    class PolicyCfg(GroupCfg):
        motion_joint_pos_command = TermCfg()
        motion_joint_vel_command = TermCfg()
        motion_ref_ori_b = TermCfg()
        # base_ang_vel = TermCfg(history_length=24)
        # joint_pos = TermCfg(history_length=24)
        # joint_vel = TermCfg(history_length=24)
        base_ang_vel = TermCfg()
        joint_pos = TermCfg()
        joint_vel = TermCfg()
        actions = TermCfg()

    policy = PolicyCfg()
