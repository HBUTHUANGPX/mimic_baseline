from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
)
from dataclasses import MISSING
from typing import Literal


@configclass  # 无特权信息的训练
class Q1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    save_interval = 500
    obs_groups = (
        {
            "policy": [
                "command_with_noise_wo_privilege",
                "proprioception_with_noise_wo_privilege",
                "last_action",
                # "policy"
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "critic": [
                "command",
                "proprioception",
                "last_action",
                # "critic"
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
            "policy_window":[
                "command_window_with_noise_wo_privilege",
            ],
            "critic_window":[
                "command_window",
            ]
        },
    )
    experiment_name = "q1_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.005,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )
@configclass  # 特权信息的训练
class Q1FlatPPOPureRunnerCfg(Q1FlatPPORunnerCfg):
    obs_groups = (
        {
            "policy": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "critic": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
        },
    ) 

@configclass
class RslRlPpoActorCriticDistillCfg(RslRlPpoActorCriticCfg):
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_obs_normalization: bool = False, 
        student_obs_normalization: bool = False, 
@configclass # 无特权信息的single FSQ蒸馏训练
class Q1FlatPPODistillSingleFSQRunnerCfg(Q1FlatPPORunnerCfg):
    obs_groups = (
        {
            "policy": [
                "command_with_noise_wo_privilege",
                "proprioception_with_noise_wo_privilege",
                "last_action",
            ],  
            "critic": [
                "command",
                "proprioception",
                "last_action",
            ], 
            "teacher": [
                "command",
                "proprioception",
                "last_action",
            ],  
            "policy_window":[
                "command_window_with_noise_wo_privilege",
            ],
            "critic_window":[
                "command_window",
            ]
        },
    )
    
    policy = RslRlPpoActorCriticDistillCfg(
        init_noise_std=0.8,
        student_obs_normalization=True,
        critic_obs_normalization=True,
        teacher_obs_normalization = True,
        student_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        teacher_hidden_dims = [512, 256, 128],
        activation="elu",
    )
    def __post_init__(self):
        super().__post_init__()
        self.class_name = "OnPolicyDisstillationRunnerFSQ"
        self.policy.class_name = "ActorCriticSingleFSQDistillation"
        self.algorithm.class_name = "PPOSingleFSQDistillation"

@configclass # 无特权信息的single FSQ训练
class Q1FlatPPOSingleFSQRunnerCfg(Q1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.class_name = "OnPolicyRunnerFSQ"
        self.policy.class_name = "ActorCriticSingleFSQ"
        self.algorithm.class_name = "PPOSingleFSQ"

@configclass  # 有特权信息WO DR 的训练
class Q1FlatTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    save_interval = 500
    obs_groups = (
        {
            "policy": [
                "command",
                "proprioception",
                "last_action",
                ],  
            "critic": [
                "command",
                "proprioception",
                "last_action",
                ], 
        },
    )
    experiment_name = "q1_flat_teacher"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.005,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )

@configclass  # 有特权信息的训练
class PureQ1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    obs_groups = (
        {
            "policy": [
                "command_with_noise",
                "proprioception_with_noise",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "critic": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
        },
    )
    save_interval = 5000
    experiment_name = "pure_q1_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# 新增 CVAE 配置类，继承原配置以兼容
@configclass
class RslRlPpoActorCritic_Distil_CVAECfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic_CVAE"
    """The policy class name. Default is ActorCritic_CVAE."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    teacher_obs_normalization: bool = False
    """Whether to normalize the observation for the teacher network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    teacher_hidden_dims: tuple[int] | list[int] = ([256, 256, 256],)  # 用于 teacher
    """The hidden dimensions of the teacher network."""

    prior_hidden_dims: tuple[int] | list[int] = ([1024, 512, 128],)  # 用于 prior
    """The hidden dimensions of the prior network."""

    encoder_hidden_dims: tuple[int] | list[int] = ([512, 256, 128],)  # 用于 encoder
    """The hidden dimensions of the encoder network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    latent_dim: int = (64,)  # CVAE 潜在空间维度
    """The latent dimension of the CVAE."""

    beta_kl: float = (0.1,)  # KL 损失权重
    """The weight of the KL loss."""

    z_scale_factor: float = (1.0,)  # z 的缩放因子
    """The scale factor of the z."""

    normalize_mu: bool = (False,)
    """Whether to normalize the mu of the CVAE."""


@configclass
class RslRlPpoActorCritic_Distil_FSQCVAECfg(RslRlPpoActorCritic_Distil_CVAECfg):
    """Configuration for the PPO actor-critic networks with FSQ quantization."""

    class_name: str = "ActorCritic_FSQ_CVAE"
    """The policy class name. Default is ActorCritic_FSQ_CVAE."""
    levels: list[int] | None = None  # FSQ levels list (length defines latent_dim)
    preserve_symmetry: bool = False  # FSQ symmetry-preserving quantization
    noise_dropout: float = 0.0  # FSQ noise dropout (training only)


@configclass
class KLSchedule:
    start: float = 0.0001
    end: float = 0.01
    start_iteration: int = 0
    end_iteration: int = 90000


@configclass  # 对有特权信息训练的教师网络进行蒸馏
class Q1FlatCVAEDistillationStudentMultiTeacherCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    obs_groups = (
        {
            "policy": [
                "command_wo_privilege",
                "proprioception_with_noise_wo_privilege",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "critic": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
            "teacher": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'teacher' 观测组，用于教师网络
            "motion_group": ["motion_group"],  # 新增 motion_group 观测组
        },
    )
    save_interval = 500
    experiment_name = "q1_flat_distillation"
    class_name: str = "MultiTeacherDistillationRunner"
    beta_kl_schedule = KLSchedule(
        start=0.0002, end=0.1, start_iteration=0, end_iteration=90000
    )
    bc_kl_coef_schedule = KLSchedule(
        start=0.01, end=0.001, start_iteration=0, end_iteration=90000
    )
    policy = RslRlPpoActorCritic_Distil_CVAECfg(
        class_name="ActorCritic_CVAE",
        init_noise_std=0.8,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256, 128],
        prior_hidden_dims=[1024, 512, 128],
        encoder_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        teacher_obs_normalization=True,
        latent_dim=64,
        beta_kl=0.0002,
        normalize_mu=False,
        z_scale_factor=0.05,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO_Distil",
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.005,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )


@configclass  # 对有特权信息训练的教师网络进行蒸馏
class Q1FlatFSQCVAEDistillationStudentMultiTeacherCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    obs_groups = (
        {
            "policy": [
                "command_wo_privilege",
                "proprioception_with_noise_wo_privilege",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "critic": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
            "teacher": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'teacher' 观测组，用于教师网络
            "motion_group": ["motion_group"],  # 新增 motion_group 观测组
        },
    )
    save_interval = 500
    experiment_name = "q1_flat_distillation"
    class_name: str = "MultiTeacherDistillationRunner"
    beta_kl_schedule = KLSchedule(
        start=0.0002, end=0.1, start_iteration=0, end_iteration=90000
    )
    bc_kl_coef_schedule = KLSchedule(
        start=0.01, end=0.001, start_iteration=0, end_iteration=90000
    )
    policy = RslRlPpoActorCritic_Distil_FSQCVAECfg(
        class_name="ActorCritic_FSQ_CVAE",
        init_noise_std=0.8,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256, 128],
        prior_hidden_dims=[1024, 512, 128],
        encoder_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        teacher_obs_normalization=True,
        levels=[5] * 32,  # 每个潜在维度的 FSQ 量化级别，长度应等于 latent_dim
        preserve_symmetry=True,  # 是否使用对称量化
        noise_dropout=0.0,  # FSQ 量化的噪声
        beta_kl=0.0002,
        normalize_mu=False,
        z_scale_factor=0.05,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO_Distil",
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.005,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )


