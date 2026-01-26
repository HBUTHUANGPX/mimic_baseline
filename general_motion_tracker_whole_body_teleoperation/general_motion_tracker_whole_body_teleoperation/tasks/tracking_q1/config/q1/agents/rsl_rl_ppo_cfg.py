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
    save_interval = 1000
    obs_groups = (
        {
            "policy": ["policy"],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "critic": ["critic"],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
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


@configclass  # 有特权信息的训练
class PureQ1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15001
    # max_iterations = 10001
    # max_iterations = 3001
    # obs_groups ={
    #     "policy": ["policy"],  # 映射到环境提供的 'policy' 观测组，用于演员网络
    #     "critic": ["critic"],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
    # },
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
    save_interval = 1000
    experiment_name = "pure_q1_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
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
class RslRlPpoActorCritic_Distil_CVAECfg():
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

    teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],  # 用于 teacher
    """The hidden dimensions of the teacher network."""

    prior_hidden_dims: tuple[int] | list[int] = [1024, 512, 128],   # 用于 prior
    """The hidden dimensions of the prior network."""

    encoder_hidden_dims: tuple[int] | list[int] = [512, 256, 128],  # 用于 encoder
    """The hidden dimensions of the encoder network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    latent_dim: int = 64,  # CVAE 潜在空间维度
    """The latent dimension of the CVAE."""

    beta_kl: float = 0.1,  # KL 损失权重
    """The weight of the KL loss."""

    z_scale_factor: float = 1.0,  # z 的缩放因子
    """The scale factor of the z."""

    normalize_mu: bool = False,  
    """Whether to normalize the mu of the CVAE."""

class RslRlDistillationStudentTeacher_CVAECfg(RslRlDistillationStudentTeacherCfg):
    latent_dim: int = 64                            # 潜在空间维度
    beta_kl: float = 0.0001                            # KL 散度权重
    class_name: str = "StudentTeacher_CVAE"         # 新类名（将在 modules 中定义）
    normalize_mu=True,  # 新参数：启用对 mu 的 EmpiricalNormalization
    z_scale_factor: float = 1.0,  # z 的缩放因子
    prior_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
    encoder_hidden_dims: tuple[int] | list[int] = [256, 256, 256],

@configclass  # 对有特权信息训练的教师网络进行蒸馏
class Q1FlatCVAEDistillationStudentMultiTeacherCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    obs_groups = (
        {
            "policy": [
                "command_wo_privilege",
                "proprioception_wo_privilege",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "critic": ["command",
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
    policy = RslRlPpoActorCritic_Distil_CVAECfg(
        class_name="ActorCritic_CVAE",
        init_noise_std=0.8,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        prior_hidden_dims = [1024, 512, 128],
        encoder_hidden_dims = [512, 256, 128],
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
class Q1FlatDistillationStudentMultiTeacherCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    obs_groups = (
        {
            "policy": [
                "command_wo_privilege",
                "proprioception_wo_privilege",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "teacher": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
            "motion_group": ["motion_group"],  # 新增 motion_group 观测组
        },
    )
    save_interval = 500
    experiment_name = "q1_flat_distillation"
    class_name: str = "MultiTeacherDistillationRunner"
    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentMultiTeacher",
        init_noise_std=0.8,
        teacher_hidden_dims=[512, 256, 128],
        student_hidden_dims=[512, 256, 128],
        activation="elu",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        learning_rate=1.0e-3,
        gradient_length=15,
        num_learning_epochs=5,
        class_name="MultiTeacherDistillation",
        max_grad_norm=1.0,
    )

@configclass  # 对有特权信息训练的教师网络进行蒸馏
class Q1FlatDistillationStudentTeacherCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    obs_groups = (
        {
            "policy": [
                "command_wo_privilege",
                "proprioception_wo_privilege",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "teacher": [
                "command",
                "proprioception",
                "last_action",
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
            "motion_group": ["motion_group"],  # 新增 motion_group 观测组
        },
    )
    save_interval = 500
    experiment_name = "q1_flat_distillation"
    class_name: str = "DistillationRunner"
    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        init_noise_std=0.8,
        teacher_hidden_dims=[512, 256, 128],
        student_hidden_dims=[512, 256, 128],
        activation="elu",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        learning_rate=1.0e-3,
        gradient_length=15,
        num_learning_epochs=5,
        class_name="Distillation",
        max_grad_norm=1.0,
    )
