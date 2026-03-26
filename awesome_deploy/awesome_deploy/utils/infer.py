"""Robot-side inference wrapper used by the MuJoCo simulator."""

import copy

import numpy as np

from awesome_deploy.inference import (
    ModelProtocol,
    RuntimeStateBuilder,
    TRANSFORM_REGISTRY,
    load_protocol_from_file,
)
from awesome_deploy.inference.backends import OnnxBackend
from awesome_deploy.inference.engine import InferenceEngine
from awesome_deploy.inference.io_adapters import ProtocolAdapter
from awesome_deploy.utils.cfg import cfg
from awesome_deploy.utils.motion_source_factory import build_motion_source
from awesome_deploy.utils.obscfg import get_obs_cfg
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
        # print("==infere init==")
        self._init_robot_conf()
        self._init_inference()
        self.pin = pin_mj(cfg)
        self.obs_cfg = get_obs_cfg()
        self.obs_manager = SimpleObservationManager(self.obs_cfg, self)
        self.first_frame_pos = self._motion_joint_pos_for_mujoco(0)

    def _init_inference(self):
        """Builds motion data and the backend-agnostic inference engine."""
        self.body_indexes = np.asarray(
            self.motion_body_names_in_isaacsim_index, dtype=np.int64
        )
        self.motion = build_motion_source(
            cfg,
            self.body_indexes.tolist(),
            "cpu",
            body_names=cfg.motion_body_names,
        )
        self.policy_dt = cfg.policy_dt
        if cfg.motion_play and not getattr(self.motion, "is_realtime", False):
            motion_fps = float(np.asarray(self.motion.fps, dtype=np.float32).reshape(-1)[0])
            if motion_fps > 0.0:
                self.policy_dt = 1.0 / motion_fps
        self.control_decimation = int(self.policy_dt / cfg.simulator_dt)
        print("control_decimation: ", self.control_decimation)
        protocol = self._load_model_protocol()
        self.inference_engine = InferenceEngine(
            backend=OnnxBackend(),
            io_adapter=ProtocolAdapter(protocol),
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
        self.runtime_state_builder = RuntimeStateBuilder(
            protocol=protocol,
            signature=signature,
        )
        action_output_name = self._get_primary_output_name(protocol)
        action_spec = signature.outputs.get(action_output_name)
        if (
            action_spec is None
            or len(action_spec.shape) < 2
            or action_spec.shape[1] is None
        ):
            raise RuntimeError(
                "Inference output 'actions' must have a fixed second dimension."
            )
        self.action_num = int(action_spec.shape[1])
        self.obs_num = self.runtime_state_builder.get_primary_observation_dim()
        self.action = np.zeros(self.action_num, dtype=np.float32)
        self.action_clip = cfg.action_clip
        self.action_scale = cfg.action_scale
        self.obs = np.zeros(self.obs_num, dtype=np.float32)
        self.single_obs = np.zeros(self.obs_num, dtype=np.float32)

    def _load_model_protocol(self) -> ModelProtocol:
        """Loads the model protocol declared next to the active policy asset.

        The protocol file is the single source of truth for model IO naming and
        buffer semantics. ``infer.py`` intentionally does not reconstruct those
        bindings in code anymore.
        """
        return load_protocol_from_file(
            cfg.protocol_path,
            TRANSFORM_REGISTRY,
        )

    def _get_primary_output_name(self, protocol: ModelProtocol) -> str:
        """Returns the raw backend output name bound as the primary action."""
        for output_name, binding in protocol.output_bindings.items():
            if binding.target_kind == "primary":
                return output_name
        raise RuntimeError("Model protocol does not define a primary output binding.")

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
        # for i in range(len(mujoco_joint_name)):
        #     print(
        #         "  - "
        #         + mujoco_joint_name[i]
        #         + ": {kp: "
        #         + str(self.P_gains[i])
        #         + ", kd: "
        #         + str(self.D_gains[i])
        #         + ", torque_max: "
        #         + str(self.tq_max[i])
        #         + ", default_pos: "
        #         + str(self.default_pos[i])
        #         + "}"
        #     )
        # print("mujoco_joint_name:\r\n", mujoco_joint_name)
        # ``isaac_sim2mujoco_index`` reorders policy-space actions into MuJoCo
        # joint order for low-level control.
        self.isaac_sim2mujoco_index = [
            cfg.isaac_sim_joint_name.index(name) for name in mujoco_joint_name
        ]
        # print("isaac_sim2mujoco_index:\r\n", self.isaac_sim2mujoco_index)
        # ``mujoco2isaac_sim_index`` performs the inverse mapping for
        # observation construction.
        self.mujoco2isaac_sim_index = [
            mujoco_joint_name.index(name) for name in cfg.isaac_sim_joint_name
        ]
        # print("mujoco2isaac_sim_index:\r\n", self.mujoco2isaac_sim_index)
        self.motion_body_names_in_isaacsim_index = [
            cfg.isaac_sim_link_name.index(name) for name in cfg.motion_body_names
        ]
        # print("motion_body_index:\r\n", self.motion_body_names_in_isaacsim_index)

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

    def _motion_joint_pos_for_policy(self, index):
        """Returns reference joint positions in policy joint order."""
        joint_pos = np.asarray(self.motion.joint_pos[index], dtype=np.float32)
        if getattr(self.motion, "joint_order_space", "isaac") == "mujoco":
            return joint_pos[..., self.mujoco2isaac_sim_index]
        return joint_pos

    def _motion_joint_pos_for_mujoco(self, index):
        """Returns reference joint positions in MuJoCo joint order."""
        joint_pos = np.asarray(self.motion.joint_pos[index], dtype=np.float32)
        if getattr(self.motion, "joint_order_space", "isaac") == "isaac":
            return joint_pos[..., self.isaac_sim2mujoco_index]
        return joint_pos

    def _motion_body_pos_for_policy(self, index):
        """Returns reference body positions in policy body order."""
        return np.asarray(self.motion.body_pos_w[index], dtype=np.float32)

    def _motion_body_quat_for_policy(self, index, body_index=None):
        """Returns reference body quaternions in policy body order."""
        body_quat = np.asarray(self.motion.body_quat_w[index], dtype=np.float32)
        if body_index is None:
            return body_quat
        return body_quat[body_index]

    def minimum_infer(self):
        """Runs one minimal policy inference step and updates target positions."""
        motion = getattr(self, "motion", None)
        if motion is not None and hasattr(motion, "advance"):
            motion.advance()
        runtime_state = self.runtime_state_builder.build(self)
        result = self.inference_engine.step(runtime_state)
        self.latest_inference = result
        if result.primary_action is None:
            raise RuntimeError("Inference result does not contain a primary action.")
        self.post_action(result.primary_action)

    def compute_obs_group(self, group_name: str) -> np.ndarray:
        """Computes one named observation group through the manager.

        Args:
            group_name: Observation group defined by the active observation
                configuration.

        Returns:
            Flattened numpy observation vector for the requested group.
        """
        group_obs = self.obs_manager.compute_group(group_name, update_history=True)
        return np.clip(np.asarray(group_obs, dtype=np.float32).reshape(-1), -10, 10)

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
