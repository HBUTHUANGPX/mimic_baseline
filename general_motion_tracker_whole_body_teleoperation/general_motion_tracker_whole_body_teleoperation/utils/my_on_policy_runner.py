from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from general_motion_tracker_whole_body_teleoperation.utils.exporter import (
    attach_onnx_metadata,
)


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb", "swanlab"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(
                self.alg.policy,
                normalizer=self.obs_normalizer,
                path=policy_path,
                filename=filename,
            )
            run_name = getattr(self.logger.writer, "run_name", None)
            attach_onnx_metadata(
                self.env.unwrapped, run_name, path=policy_path, filename=filename
            )
            self.logger.writer.save_file(policy_path + filename)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        registry_name: str = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb", "swanlab"]:
            # link the artifact registry to this run
            if self.logger_type == "wandb" and self.registry_name is not None:
                import wandb

                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
