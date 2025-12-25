from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
)


@configclass  # 无特权信息的训练
class Q1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    save_interval = 1500
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
    max_iterations = 90001
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
    save_interval = 1500
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
class RslRlDistillationStudentTeacher_CVAECfg(RslRlDistillationStudentTeacherCfg):
    latent_dim: int = 64                            # 潜在空间维度
    beta_kl: float = 0.1                            # KL 散度权重
    class_name: str = "StudentTeacher_CVAE"         # 新类名（将在 modules 中定义）
    normalize_mu=True,  # 新参数：启用对 mu 的 EmpiricalNormalization
    z_scale_factor: float = 1.0,  # z 的缩放因子

@configclass  # 对有特权信息训练的教师网络进行蒸馏
class Q1FlatDistillationStudentTeacherCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 90001
    obs_groups = (
        {
            "policy": [
                "command_with_noise_wo_privilege",
                "proprioception_with_noise_wo_privilege",
                "last_action",
            ],  # 映射到环境提供的 'policy' 观测组，用于演员网络
            "teacher": [
                "command_with_noise",
                "proprioception_with_noise",
                "last_action",
            ],  # 映射到环境提供的 'critic' 观测组，用于评论家网络
        },
    )
    save_interval = 1500
    experiment_name = "q1_flat_distillation"
    policy = RslRlDistillationStudentTeacher_CVAECfg(
        class_name="StudentTeacher_CVAE",
        init_noise_std=0.8,
        teacher_hidden_dims=[512, 256, 128],
        student_hidden_dims=[512, 256, 128],
        activation="elu",
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        latent_dim=8,
        beta_kl=0.1,
        normalize_mu=True,
        z_scale_factor=0.05,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        learning_rate=1.0e-3,
        gradient_length=15,
        num_learning_epochs=5,
        class_name="Distillation",
    )
