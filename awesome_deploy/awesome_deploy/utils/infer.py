"""Robot-side inference wrapper used by the MuJoCo simulator."""

import copy

import numpy as np

from awesome_deploy.inference import InferenceContext
from awesome_deploy.inference.backends import OnnxBackend
from awesome_deploy.inference.engine import InferenceEngine
from awesome_deploy.inference.io_adapters import DefaultMimoAdapter
from awesome_deploy.utils.cfg import cfg
from awesome_deploy.utils.motion_loader import MotionLoader
from awesome_deploy.utils.obscfg import ObsCfg
from awesome_deploy.utils.observation_manager import SimpleObservationManager
from awesome_deploy.utils.pinocchio_func import pin_mj


class infere:
    """Bridges simulator state, observations, and the policy inference engine.

    This class still owns robot-specific preprocessing and postprocessing:

    - It prepares joint-order conversion tables.
    - It computes policy observations from simulator state.
    - It converts policy actions into MuJoCo PD targets.

    Backend-specific model execution is delegated to ``InferenceEngine``.
    """

    def __init__(self):
        """Initializes robot state, inference engine, and observation manager."""
        print("==infere init==")
        self._init_robot_conf()
        self._init_inference()
        self.pin = pin_mj(cfg)
        self.obs_manager = SimpleObservationManager(ObsCfg(), self)
        self.first_frame_pos = np.copy(self.motion.joint_pos[0])[
            self.isaac_sim2mujoco_index
        ]

    def _init_inference(self):
        """Builds motion data and the backend-agnostic inference engine."""
        self.body_indexes = np.asarray(
            self.motion_body_names_in_isaacsim_index, dtype=np.int64
        )
        self.motion = MotionLoader(
            cfg.motion_file,
            self.body_indexes,
            "cpu",
        )
        self.policy_dt = cfg.policy_dt
        if cfg.motion_play:
            self.policy_dt = (1 / self.motion.fps)[0]
        else:
            self.policy_dt = cfg.policy_dt
        self.control_decimation = int(self.policy_dt / cfg.simulator_dt)
        print("control_decimation: ", self.control_decimation)
        # Keep the current ONNX implementation behind a generic engine so the
        # simulator no longer depends on any concrete backend API.
        self.inference_engine = InferenceEngine(
            backend=OnnxBackend(),
            io_adapter=DefaultMimoAdapter(),
            model_path=cfg.policy_path,
            device="cpu",
        )
        self.inference_engine.load()
        self.latest_inference = None
        signature = self.inference_engine.signature
        if signature is None:
            raise RuntimeError(
                "Inference engine signature is not available after load."
            )
        action_spec = signature.outputs.get("actions")
        if (
            action_spec is None
            or len(action_spec.shape) < 2
            or action_spec.shape[1] is None
        ):
            raise RuntimeError(
                "Inference output 'actions' must have a fixed second dimension."
            )
        obs_spec = signature.inputs.get("obs")
        if obs_spec is None or len(obs_spec.shape) < 2 or obs_spec.shape[1] is None:
            raise RuntimeError(
                "Inference input 'obs' must have a fixed second dimension."
            )
        self.action_num = int(action_spec.shape[1])
        self.obs_num = int(obs_spec.shape[1])
        self.action = np.zeros(self.action_num, dtype=np.float32)
        self.action_clip = cfg.action_clip
        self.action_scale = cfg.action_scale
        self.obs = np.zeros(self.obs_num, dtype=np.float32)
        self.single_obs = np.zeros(self.obs_num, dtype=np.float32)

    def _init_robot_conf(self):
        """Initializes flattened motor parameters and name mapping indices."""
        self.default_pos = np.array(
            [value for part in cfg.motor_cfg.values() for value in part["default_pos"]],
            dtype=np.float32,
        )
        self.P_gains = np.array(
            [value for part in cfg.motor_cfg.values() for value in part["stiffness"]],
            dtype=np.float32,
        )
        self.D_gains = np.array(
            [value for part in cfg.motor_cfg.values() for value in part["damping"]],
            dtype=np.float32,
        )
        self.tq_max = np.array(
            [value for part in cfg.motor_cfg.values() for value in part["torque_max"]],
            dtype=np.float32,
        )
        self.P_n = np.zeros_like(self.default_pos)
        self.V_n = np.zeros_like(self.default_pos)
        self.target_dof_pos = np.zeros_like(self.default_pos)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        mujoco_joint_name = cfg.urdf_graph.joint_order_by_file()
        for i in range(len(mujoco_joint_name)):
            print(
                "  - "
                + mujoco_joint_name[i]
                + ": {kp: "
                + str(self.P_gains[i])
                + ", kd: "
                + str(self.D_gains[i])
                + ", torque_max: "
                + str(self.tq_max[i])
                + ", default_pos: "
                + str(self.default_pos[i])
                + "}"
            )
        print("mujoco_joint_name:\r\n", mujoco_joint_name)
        # ``isaac_sim2mujoco_index`` reorders policy-space actions into MuJoCo
        # joint order for low-level control.
        self.isaac_sim2mujoco_index = [
            cfg.isaac_sim_joint_name.index(name) for name in mujoco_joint_name
        ]
        print("isaac_sim2mujoco_index:\r\n", self.isaac_sim2mujoco_index)
        # ``mujoco2isaac_sim_index`` performs the inverse mapping for
        # observation construction.
        self.mujoco2isaac_sim_index = [
            mujoco_joint_name.index(name) for name in cfg.isaac_sim_joint_name
        ]
        print("mujoco2isaac_sim_index:\r\n", self.mujoco2isaac_sim_index)
        self.motion_body_names_in_isaacsim_index = [
            cfg.isaac_sim_link_name.index(name) for name in cfg.motion_body_names
        ]
        print("motion_body_index:\r\n", self.motion_body_names_in_isaacsim_index)

    @property
    def time_step(self) -> int:
        """Returns the logical rollout step stored in the inference engine."""
        if hasattr(self, "inference_engine"):
            return int(self.inference_engine.buffers.get("time_step", 1))
        return 1

    @time_step.setter
    def time_step(self, value: int) -> None:
        """Updates the logical rollout step inside the inference engine."""
        if hasattr(self, "inference_engine"):
            self.inference_engine.buffers.set("time_step", int(value))

    def post_action(self, action):
        """Converts policy action space into MuJoCo joint target positions.

        Args:
            action: Raw policy action vector in policy joint order.
        """
        self.action[:] = np.asarray(action, dtype=np.float32).reshape(-1)
        # Reorder into MuJoCo joint order, then transform the normalized policy
        # output into a joint-space target compatible with the PD controller.
        action = (
            np.clip(
                copy.deepcopy(self.action[self.isaac_sim2mujoco_index]),
                -self.action_clip,
                self.action_clip,
            )
            * self.action_scale
            * self.tq_max
            / self.P_gains
            + self.default_pos
        )
        target_q = action.clip(-self.action_clip, self.action_clip)
        self.target_dof_pos = target_q

    def minimum_infer(self):
        """Runs one minimal policy inference step and updates target positions."""
        obs = self.update_obs()
        context = InferenceContext(
            obs=obs,
            time_step=self.time_step,
            command=self.cmd,
            extras={"motion": getattr(self, "motion", None)},
        )
        result = self.inference_engine.step(context)
        self.latest_inference = result
        if result.primary_action is None:
            raise RuntimeError("Inference result does not contain a primary action.")
        self.post_action(result.primary_action)

    def update_obs(self):
        """Computes and clips the current policy observation vector.

        Returns:
            Flattened observation vector in the order specified by ``ObsCfg``.
        """
        self.obs = np.clip(
            self.obs_manager.compute_group("policy", update_history=True), -10, 10
        )
        return self.obs

    def _obs_motion_joint_pos_command(self):
        """Returns the reference motion joint position term.

        Raises:
            NotImplementedError: Implemented by the simulator subclass.
        """
        raise NotImplementedError

    def _obs_motion_joint_vel_command(self):
        """Returns the reference motion joint velocity term."""
        raise NotImplementedError

    def _obs_motion_ref_ori_b(self):
        """Returns the body-frame orientation error to the motion reference."""
        raise NotImplementedError

    def _obs_base_ang_vel(self):
        """Returns the robot base angular velocity observation."""
        raise NotImplementedError

    def _obs_joint_pos(self):
        """Returns the current robot joint position observation."""
        raise NotImplementedError

    def _obs_joint_vel(self):
        """Returns the current robot joint velocity observation."""
        raise NotImplementedError

    def _obs_actions(self):
        """Returns the previous policy action observation."""
        raise NotImplementedError
