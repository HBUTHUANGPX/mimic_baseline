from deploy.utils.observation_manager import SimpleObservationManager, TermCfg, GroupCfg
class ObsCfg:
    """观测总配置：每个属性是一个 GroupCfg 实例。"""

    class PolicyCfg(GroupCfg):
        ref_human_anchor_rot6d_in_sim_anchor = TermCfg()
        sim_robot_anchor_rot6d_w = TermCfg(history_length=8)
        base_ang_vel = TermCfg(history_length=8)
        joint_pos = TermCfg(history_length=8)
        joint_vel = TermCfg(history_length=8)
        actions = TermCfg()
    class HumanFSQCfg(GroupCfg):
        actor_ref_human_fsq_feature_window = TermCfg()
    class RobotFSQCfg(GroupCfg):
        actor_ref_robot_fsq_feature_window = TermCfg()

    class RobotTokenCfg(GroupCfg):
        actor_robot_token = TermCfg()

    class HumanTokenCfg(GroupCfg):
        actor_human_token = TermCfg()
    actor_obs = PolicyCfg()
    human_obs = HumanFSQCfg()
    robot_obs = RobotFSQCfg()
    human_token_obs = RobotTokenCfg()
    robot_token_obs = HumanTokenCfg()
    