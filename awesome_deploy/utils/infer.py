from awesome_deploy.utils.observation_manager import SimpleObservationManager, TermCfg, GroupCfg
from awesome_deploy.utils.motion_loader import MotionLoader
from awesome_deploy.utils.obscfg import ObsCfg
from awesome_deploy.utils.cfg import cfg,current_path
from awesome_deploy.utils.pinocchio_func import pin_mj
import copy
import onnxruntime as ort
import numpy as np

class infere:
    policy: ort.InferenceSession
    def __init__(self):
        print("==infere init==")
        self._init_robot_conf()
        self._init_policy_conf()
        self.pin = pin_mj(cfg)
        self.obs_manager = SimpleObservationManager(ObsCfg(), self)
        self.first_frame_pos = np.copy(self.motion.joint_pos[0])[self.isaac_sim2mujoco_index]

    def _init_policy_conf(self):
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
        self.policy = self.load_onnx_model(cfg.policy_path)
        if hasattr(self.policy, "_outputs_meta"):
            for (idx,meta) in enumerate(self.policy._outputs_meta):
                if meta.name == "actions":
                    self.action_num = meta.shape[1]
        if hasattr(self.policy, "_inputs_meta"):
            for (idx,meta) in enumerate(self.policy._inputs_meta):
                if meta.name == "obs":
                    self.obs_num = meta.shape[1]
        self.h2_action = np.zeros(self.action_num, dtype=np.float32)
        self.h_action = np.zeros(self.action_num, dtype=np.float32)
        self.action = np.zeros(self.action_num, dtype=np.float32)
        self.action_clip = cfg.action_clip

        self.action_scale = cfg.action_scale
        self.action_num = self.action_num
        self.obs = np.zeros(self.obs_num * cfg.frame_stack, dtype=np.float32)
        self.time_step = 1
        self.single_obs = np.zeros(self.obs_num, dtype=np.float32)

    def _init_robot_conf(self):
        self.default_pos = np.array(
            cfg.leg_default_pos
            + cfg.pelvis_default_pos
            + cfg.arm_default_pos,
            dtype=np.float32,
        )  # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.P_gains = np.array(
            cfg.leg_P_gains + cfg.pelvis_P_gains + cfg.arm_P_gains,
        )  # [70.0, 70.0, 3.0, 70.0, 70.0, 70.0, 1.5, 180.0, 180.0, 70.0, 70.0, 180.0, 180.0, 70.0, 70.0, 330.0, 330.0, 20.0, 20.0, 330.0, 330.0, 20.0, 20.0, 70.0, 70.0, 20.0, 20.0, 70.0, 70.0]
        self.D_gains = np.array(
            cfg.leg_D_gains + cfg.pelvis_D_gains + cfg.arm_D_gains,
        )  # [1.5, 1.5, 0.6, 1.5, 1.5, 1.5, 0.3, 2.5, 2.5, 2.0, 2.0, 2.5, 2.5, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5]
        self.tq_max = np.array(
            cfg.leg_tq_max + cfg.pelvis_tq_max + cfg.arm_tq_max,
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
        self.isaac_sim2mujoco_index = [
            cfg.isaac_sim_joint_name.index(name) for name in mujoco_joint_name
        ]
        print("isaac_sim2mujoco_index:\r\n", self.isaac_sim2mujoco_index)
        self.mujoco2isaac_sim_index = [
            mujoco_joint_name.index(name) for name in cfg.isaac_sim_joint_name
        ]
        print("mujoco2isaac_sim_index:\r\n", self.mujoco2isaac_sim_index)

        self.motion_body_names_in_isaacsim_index = [
            cfg.isaac_sim_link_name.index(name) for name in cfg.motion_body_names
        ]
        print("motion_body_index:\r\n", self.motion_body_names_in_isaacsim_index)

    def load_onnx_model(self, onnx_path, device="cpu"):
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session

    def run_onnx_inference(self, session, obs, time_step):
        # 转换为numpy array并确保数据类型正确
        obs = np.asarray(obs)
        time_step = np.asarray(time_step, dtype=np.float32)
        # 获取输入名称
        obs_name = session.get_inputs()[0].name
        time_step_name = session.get_inputs()[1].name
        # 运行推理
        (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = session.run(
            None,
            {
                obs_name: obs.reshape(1, self.obs_num),
                time_step_name: time_step.reshape(1, 1),
            },
        )
        return (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        )  # 默认返回第一个输出

    def _policy_reasoning(self):

        (
            act,
            self.r_joint_pos,
            self.r_joint_vel,
            self.r_body_pos_w,
            self.r_body_quat_w,
            self.r_body_lin_vel_w,
            self.r_body_ang_vel_w,
        ) = self.run_onnx_inference(
            self.policy, self.obs.astype(np.float32), self.time_step
        )
        self.action[:] = act.copy()

    def post_action(self):
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
        # print(target_q)
        self.target_dof_pos = target_q  # + self.default_pos[: self.action_num]

    def minimum_infer(self):
        self.update_obs()
        self.h2_action = self.h_action.copy()
        self.h_action = self.action.copy()
        self._policy_reasoning()
        self.post_action()
        self.time_step+=1
        # self.time_step=0

    def update_obs(self):
        """
        +----------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (154,)) |
        +------------+--------------------------------+------------+
        |   Index    | Name                           |   Shape    |
        +------------+--------------------------------+------------+
        |     0      | command                        |   (58,)    |
        |     1      | motion_ref_ori_b               |    (6,)    |
        |     2      | base_ang_vel                   |    (3,)    |
        |     3      | joint_pos                      |   (29,)    |
        |     4      | joint_vel                      |   (29,)    |
        |     5      | actions                        |   (29,)    |
        +------------+--------------------------------+------------+
        """
        self.obs = np.clip(
            self.obs_manager.compute_group("policy", update_history=True), -10, 10
        )

    def _obs_motion_joint_pos_command(self):
        raise NotImplementedError

    def _obs_motion_joint_vel_command(self):
        raise NotImplementedError

    def _obs_motion_ref_ori_b(self):
        raise NotImplementedError

    def _obs_base_ang_vel(self):
        raise NotImplementedError

    def _obs_joint_pos(self):
        raise NotImplementedError

    def _obs_joint_vel(self):
        raise NotImplementedError

    def _obs_actions(self):
        raise NotImplementedError

