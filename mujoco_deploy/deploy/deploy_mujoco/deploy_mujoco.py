from deploy.utils.video_recorder import VideoRecorder
from deploy.utils.math_func import *
from deploy.utils.cfg import cfg, current_path
from deploy.utils.infer import infere

import numpy as np
import time
import os

import mujoco.viewer
import mujoco

np.set_printoptions(precision=16, linewidth=100, threshold=np.inf, suppress=True)


class simulator(infere):

    def __init__(self):
        # Load robot model
        self.spec = mujoco.MjSpec.from_file(cfg.mjcf_path)
        self.m = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
        self.m.opt.timestep = cfg.simulator_dt
        self.d = mujoco.MjData(self.m)
        self._scene = mujoco.MjvScene(self.m, 100000)
        print(f"Number of actuators: {self.m.nu}")

        self._init_robot_conf()
        super().__init__()

        self.paused = False
        self.change_id = 0
        self.show_human_skeleton = True
        self.show_human_skeleton_axes = True
        self.human_parent_indices = self._load_human_parent_indices()
        self.vel_geom_id = None
        self._persistent_user_geom_count = 0
        self.video_recorder = VideoRecorder(
            path=current_path + "/deploy_mujoco_recordings",
            tag=None,
            video_name="video_0",
            fps=int(1 / cfg.policy_dt),
            compress=False,
        )
        self.data_save = []
        self.first_flag = True

    def _load_human_parent_indices(self):
        motion_files = getattr(self.motion, "motion_file", None)
        if not motion_files:
            return self._fallback_human_parent_indices()

        with np.load(motion_files[0], allow_pickle=True) as data:
            if "human_parent_indices" not in data or "human_joint_names" not in data:
                return self._fallback_human_parent_indices()
            human_joint_names = data["human_joint_names"].tolist()
            source_parent_indices = np.asarray(
                data["human_parent_indices"], dtype=np.int32
            )
        desired_names = cfg.desire_human_joint_names
        desired_name_set = set(desired_names)
        desired_parent_indices = np.full(len(desired_names), -1, dtype=np.int32)

        for desired_idx, name in enumerate(desired_names):
            source_idx = human_joint_names.index(name)
            parent_idx = int(source_parent_indices[source_idx])
            while parent_idx >= 0:
                parent_name = human_joint_names[parent_idx]
                if parent_name in desired_name_set:
                    desired_parent_indices[desired_idx] = desired_names.index(parent_name)
                    break
                parent_idx = int(source_parent_indices[parent_idx])
        return desired_parent_indices

    def _fallback_human_parent_indices(self):
        parent_by_name = {
            "Hips": -1,
            "Spine1": "Hips",
            "Spine2": "Spine1",
            "Chest": "Spine2",
            "Neck1": "Chest",
            "Neck2": "Neck1",
            "Head": "Neck2",
            "HeadEnd": "Head",
            "LeftShoulder": "Chest",
            "LeftArm": "LeftShoulder",
            "LeftForeArm": "LeftArm",
            "LeftHand": "LeftForeArm",
            "RightShoulder": "Chest",
            "RightArm": "RightShoulder",
            "RightForeArm": "RightArm",
            "RightHand": "RightForeArm",
            "LeftLeg": "Hips",
            "LeftShin": "LeftLeg",
            "LeftFoot": "LeftShin",
            "LeftToeBase": "LeftFoot",
            "LeftToeEnd": "LeftToeBase",
            "RightLeg": "Hips",
            "RightShin": "RightLeg",
            "RightFoot": "RightShin",
            "RightToeBase": "RightFoot",
            "RightToeEnd": "RightToeBase",
        }
        parent_indices = np.full(len(cfg.desire_human_joint_names), -1, dtype=np.int32)
        for idx, name in enumerate(cfg.desire_human_joint_names):
            parent_name = parent_by_name.get(name, -1)
            if parent_name != -1 and parent_name in cfg.desire_human_joint_names:
                parent_indices[idx] = cfg.desire_human_joint_names.index(parent_name)
        return parent_indices

    def draw_sphere(self, scene, position, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=np.asarray(position, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def draw_line(self, scene, start, end, width, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3, dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=width,
            from_=np.asarray(start, dtype=np.float64),
            to=np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1

    def draw_axes(self, scene, position, rotation, axis_length=0.06, axis_width=0.003):
        rot_mat = matrix_from_quat(np.asarray(rotation, dtype=np.float64))
        colors = (
            np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        )
        for axis_idx, color in enumerate(colors):
            self.draw_line(
                scene,
                position,
                position + rot_mat[:, axis_idx] * axis_length,
                axis_width,
                color,
            )

    def draw_human_skeleton(
        self,
        positions,
        rotations=None,
        parent_indices=None,
        show_axes=False,
        joint_radius=0.025,
        bone_width=0.008,
    ):
        scene = self.viewer.user_scn
        scene.ngeom = self._persistent_user_geom_count
        parent_indices = (
            self.human_parent_indices if parent_indices is None else parent_indices
        )
        joint_rgba = np.array([1.0, 0.8, 0.1, 0.9], dtype=np.float32)
        bone_rgba = np.array([0.3, 0.9, 1.0, 0.7], dtype=np.float32)

        for joint_idx, position in enumerate(np.asarray(positions)):
            if scene.ngeom >= scene.maxgeom:
                break
            self.draw_sphere(scene, position, joint_radius, joint_rgba)
            parent_idx = int(parent_indices[joint_idx])
            if parent_idx >= 0:
                self.draw_line(
                    scene, positions[parent_idx], position, bone_width, bone_rgba
                )
            if show_axes and rotations is not None and scene.ngeom + 3 < scene.maxgeom:
                self.draw_axes(scene, position, rotations[joint_idx])

    def draw_current_human_skeleton(self):
        if not self.show_human_skeleton or not hasattr(self.motion, "human_body_pos_w"):
            return
        frame_idx = min(int(self.time_step), self.motion.human_body_pos_w.shape[0] - 1)
        rotations = None
        if hasattr(self.motion, "human_body_quat_w"):
            rotations = self.motion.human_body_quat_w[frame_idx]
        self.draw_human_skeleton(
            self.motion.human_body_pos_w[frame_idx],
            rotations=rotations,
            show_axes=self.show_human_skeleton_axes,
        )

    def motion_play(self):
        t = int(self.time_step)
        self.d.qpos[0:3] = np.asarray(self.motion.body_pos_w[t, 0, :])
        # self.d.qpos[0:2] = 0
        # self.d.qpos[2] = 1.0
        q = np.asarray(self.motion.body_quat_w[t, 0, :])
        self.d.qpos[3:7] = q
        self.d.qpos[7 : 7 + len(self.default_pos)] = np.asarray(
            self.motion.joint_pos[t][self.isaac_sim2mujoco_index]
        )
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        self.time_step += 1
        mujoco_body_names_indices = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in cfg.motion_body_names
        ]
        body_positions = np.copy(self.d.xpos[mujoco_body_names_indices, :])
        motion_body_positions = self.motion.body_pos_w[t]

        motion_reference_body_quat = self.d.xquat[
            mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_BODY, cfg.motion_reference_body
            ),
            :,
        ]
        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        _quat = self.pin.get_link_quaternion(cfg.motion_reference_body)
        
        return

    def _init_robot_conf(self):
        super()._init_robot_conf()
        self.mujoco_all_body_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.m.nbody)
        ][1:]
        self.mujoco_body_names_indices = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.mujoco_all_body_names
        ]
        print("mujoco_all_body_names:\r\n", self.mujoco_all_body_names)
        self.fsq_human_body_indexes = [
            cfg.desire_human_joint_names.index(name)
            for name in cfg.fsq_human_body_names
        ]
        self.human_anchor_body_index = cfg.desire_human_joint_names.index(
            cfg.human_anchor_name
        )

    def run(self):
        save_data_flag = 1
        self.counter = 0
        self.target_dof_pos = self.default_pos.copy()[: self.action_num]
        self.phase = 0
        # self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        if save_data_flag:
            i = 0
            if os.path.exists("data.csv"):
                os.remove("data.csv")
        self.viewer = mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.key_callback
        )
        self.renderer = mujoco.Renderer(self.m, height=480, width=640)
        self.init_vel_geom(
            "Goal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z:{:.2f}".format(
                self.cmd[0], self.cmd[1], self.cmd[2], 0.0
            )
        )
        self.prev_qpos = self.d.qpos

        first_flag = False

        log = {
            "fps": [50],
            "dof_names": [joint.name for joint in self.spec.joints][1:],
            "body_names": self.mujoco_all_body_names,
            "dof_positions": [],
            "dof_velocities": [],
            "dof_torque": [],
            "body_positions": [],
            "body_rotations": [],
            "body_linear_velocities": [],
            "body_angular_velocities": [],
            "qpos": [],
            "qvel": [],
            "xpos": [],
            "xquat": [],
            "cvel": [],
            "P_gain": [self.P_gains],
            "D_gain": [self.D_gains],
            "target_pos": [],
            "qfrc_actuator": [],
        }

        while self.viewer.is_running():
            if not first_flag:
                first_flag = True
                if cfg.motion_play:
                    self.motion_play()
                    self.time_step = 0
                else:
                    self.motion_play()
                    ...
                mujoco.mj_step(self.m, self.d)
                self.viewer.sync()
            self.perpare_data()
            self.policy_loop()
            # print(self.time_step, self.motion.time_step_total)
            log["dof_positions"].append(np.copy(self.d.qpos[7:]))
            log["dof_velocities"].append(np.copy(self.d.qvel[6:]))
            log["dof_torque"].append(np.copy(self.d.qfrc_actuator[6:]))
            log["body_positions"].append(
                np.copy(self.d.xpos[self.mujoco_body_names_indices, :])
            )
            log["body_rotations"].append(
                np.copy(self.d.xquat[self.mujoco_body_names_indices, :])
            )
            log["body_linear_velocities"].append(
                np.copy(self.d.cvel[self.mujoco_body_names_indices, 0:3])
            )
            log["body_angular_velocities"].append(
                np.copy(self.d.cvel[self.mujoco_body_names_indices, 3:6])
            )
            log["qpos"].append(np.copy(self.d.qpos))
            log["qvel"].append(np.copy(self.d.qvel))
            log["xpos"].append(np.copy(self.d.xpos[self.mujoco_body_names_indices, :]))
            log["xquat"].append(
                np.copy(self.d.xquat[self.mujoco_body_names_indices, :])
            )
            log["cvel"].append(np.copy(self.d.cvel[self.mujoco_body_names_indices, :]))
            log["target_pos"].append(np.copy(self.target_dof_pos))
            log["qfrc_actuator"].append(np.copy(self.d.qfrc_actuator))
            # if self.time_step >= 50*60:
            if self.time_step >= self.motion.time_step_total:
                self.time_step = 0
                # break
        for k in (
            "dof_positions",
            "dof_velocities",
            "body_positions",
            "body_rotations",
            "body_linear_velocities",
            "body_angular_velocities",
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "cvel",
            "qfrc_actuator",
        ):
            log[k] = np.stack(log[k], axis=0)
        save_path = current_path + "/tmp/motion.npz"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **log)
        print("stop")
        self.video_recorder.stop()

    def perpare_data(self):
        if self.first_flag:
            self.first_flag = False
            self.init_ref_human_anchor_quat_w = self.motion.human_body_quat_w[
            0, self.human_anchor_body_index, :]
            q = self.init_ref_human_anchor_quat_w / np.linalg.norm(self.init_ref_human_anchor_quat_w)
            w, x, y, z = q
            self.init_ref_human_anchor_quat_w_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            # half = -0.5 * -np.pi/2
            half = 0
            # half = -0.5 * self.init_ref_human_anchor_quat_w_yaw
            self.yaw_comp_quat_ref_human_anchor_quat_w = np.array(
                [np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32
            )

    def policy_loop(self):
        # print("="*(20))
        self.counter += 1
        # print(self.d.qvel[0])
        self.qpos = self.d.qpos[7:]
        self.P_n = self.qpos - self.default_pos
        self.V_n = self.d.qvel[6:]

        # if self.time_step >= 100:
        if self.time_step >= self.motion.time_step_total-self.motion.future_frames:
            self.time_step = self.motion.history_frames

        if cfg.motion_play:
            self.motion_play()
        else:
            self.minimum_infer()
            # print(f"time_step: {self.time_step}")
            self.sim_loop()
        self.contact_force()
        # 更新 Renderer 场景，使用查看器的相机和选项，使图像与窗口一致
        self.renderer.update_scene(
            self.d,
            camera=self.viewer.cam,  # 使用查看器的相机视图
            scene_option=self.viewer.opt,  # 使用查看器的渲染选项
        )

        # 捕获图像：返回 (height, width, 3) 的 uint8 NumPy 数组 (RGB)
        img = self.renderer.render()
        self.video_recorder(img)

        self.draw_current_human_skeleton()
        self.viewer.sync()
        self.update_vel_geom()

    def _obs_actor_robot_token(self):
        return self.motion.actor_q_robot[int(self.time_step)]
    
    def _obs_actor_human_token(self):
        return self.motion.actor_q_human[int(self.time_step)]
     
    def _obs_ref_human_anchor_rot6d_in_sim_anchor(self):
        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        _quat = self.pin.get_link_quaternion(cfg.motion_reference_body)
        sim_robot_anchor_quat_w = np.expand_dims(_quat, axis=0)  # shape [n,4]
        ref_human_anchor_quat_w = self.motion.human_body_quat_w[
            int(self.time_step),
            self.human_anchor_body_index,
            :,
        ]  # shape [n,4]
        ref_human_anchor_quat_w = quat_mul(
            self.yaw_comp_quat_ref_human_anchor_quat_w, ref_human_anchor_quat_w
        )
        q01 = sim_robot_anchor_quat_w
        q02 = ref_human_anchor_quat_w
        if q02 is not None and q02.ndim == 1:
            q02 = np.expand_dims(q02, axis=0)
        _, ref_human_anchor_quat_in_sim_anchor = subtract_frame_transforms(
            np.zeros((1, 3), dtype=np.float32),
            q01,
            np.zeros((1, 3), dtype=np.float32),
            q02,
        )
        mat = matrix_from_quat(ref_human_anchor_quat_in_sim_anchor)
        motion_ref_ori_b = mat[..., :2].reshape(mat.shape[0], -1)  # shape [n,6]
        return motion_ref_ori_b

    def _obs_sim_robot_anchor_rot6d_w(self):
        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        _quat = self.pin.get_link_quaternion(cfg.motion_reference_body)
        sim_robot_anchor_quat_w = np.expand_dims(_quat, axis=0)
        return rot6d_from_quat(sim_robot_anchor_quat_w)
        # ref_quat_w = self.motion.body_quat_w[
        #     int(self.time_step),
        #     cfg.motion_body_names.index(cfg.motion_reference_body),
        #     :,
        # ]
        # return rot6d_from_quat(ref_quat_w)

    def _obs_base_ang_vel(self):
        return self.d.qvel[3:6]

    def _obs_joint_pos(self):
        return (self.d.qpos[7:] - self.default_pos)[self.mujoco2isaac_sim_index]

    def _obs_joint_vel(self):
        return self.d.qvel[6:][self.mujoco2isaac_sim_index]

    def _obs_actions(self):
        return self.action

    def _obs_actor_ref_robot_fsq_feature_window(self):
        start = int(self.time_step)
        end = int(self.time_step) + self.motion.window_size
        num_envs = 1
        window_size = self.motion.window_size
        motion_anchor_body_index = cfg.motion_body_names.index(
            cfg.motion_reference_body
        )
        robot_anchor_quat = self.motion.body_quat_w[
            start:end, motion_anchor_body_index
        ][None, ...]
        robot_anchor_rot6d = rot6d_from_quat(robot_anchor_quat)
        robot_anchor_pos = self.motion.body_pos_w[
            start:end, motion_anchor_body_index
        ][None, ...]
        robot_joint_pos = self.motion.joint_pos[start:end][None, ...]
        robot_body_pos = self.motion.body_pos_w[start:end][None, ...]
        robot_body_quat = self.motion.body_quat_w[start:end][None, ...]
        num_robot_bodies = robot_body_pos.shape[2]
        robot_anchor_pos_repeat = np.broadcast_to(
            robot_anchor_pos[:, :, None, :],
            (num_envs, window_size, num_robot_bodies, 3),
        )
        robot_anchor_quat_repeat = np.broadcast_to(
            robot_anchor_quat[:, :, None, :],
            (num_envs, window_size, num_robot_bodies, 4),
        )
        ref_robot_body_pos_in_ref_anchor, ref_robot_body_quat_in_ref_anchor = (
            subtract_frame_transforms(
                robot_anchor_pos_repeat.reshape(-1, 3),
                robot_anchor_quat_repeat.reshape(-1, 4),
                robot_body_pos.reshape(-1, 3),
                robot_body_quat.reshape(-1, 4),
            )
        )
        ref_robot_body_pos_in_ref_anchor = ref_robot_body_pos_in_ref_anchor.reshape(
            num_envs, window_size, -1
        )
        ref_robot_body_rot6d_in_ref_anchor = rot6d_from_quat(
            ref_robot_body_quat_in_ref_anchor
        ).reshape(num_envs, window_size, -1)
        actor_robot_feature = np.concatenate(
            (
                robot_anchor_rot6d,
                robot_joint_pos,
                # ref_robot_body_pos_in_ref_anchor,
                # ref_robot_body_rot6d_in_ref_anchor,
            ),
            axis=-1,
        )
        actor_ref_robot_fsq_feature_window = actor_robot_feature.reshape(-1)
        return actor_ref_robot_fsq_feature_window

    def _obs_actor_ref_human_fsq_feature_window(self):
        start = int(self.time_step)
        end = int(self.time_step) + self.motion.window_size
        num_envs = 1
        window_size = self.motion.window_size
        num_human_bodies = len(self.fsq_human_body_indexes)
        human_anchor_quat = self.motion.human_body_quat_w[
            start:end, self.human_anchor_body_index
        ][None, ...]
        human_anchor_rot6d = rot6d_from_quat(human_anchor_quat)
        human_anchor_pos = self.motion.human_body_pos_w[
            start:end, self.human_anchor_body_index
        ][None, ...]
        human_body_pos = self.motion.human_body_pos_w[start:end][
            :, self.fsq_human_body_indexes, :
        ][None, ...]
        human_body_quat = self.motion.human_body_quat_w[start:end][
            :, self.fsq_human_body_indexes, :
        ][None, ...]
        human_joint_quat = self.motion.human_joint_quat[start:end][
            :, self.fsq_human_body_indexes, :
        ][None, ...]
        ref_human_body_pos_from_ref_anchor_w = human_body_pos - human_anchor_pos[
            :, :, None, :
        ]
        human_anchor_quat_w = np.broadcast_to(
            human_anchor_quat[:, :, None, :],
            (num_envs, window_size, num_human_bodies, 4),
        )
        ref_human_body_pos_in_ref_anchor = quat_apply_inverse(
            human_anchor_quat_w.reshape(-1, 4),
            ref_human_body_pos_from_ref_anchor_w.reshape(-1, 3),
        ).reshape(num_envs, window_size, -1)
        ref_human_body_quat_in_ref_anchor = quat_mul(
            quat_inv(human_anchor_quat_w.reshape(-1, 4)),
            human_body_quat.reshape(-1, 4),
        )
        ref_human_body_rot6d_in_ref_anchor = rot6d_from_quat(
            ref_human_body_quat_in_ref_anchor
        ).reshape(num_envs, window_size, -1)
        human_joint_rot6d = rot6d_from_quat(human_joint_quat).reshape(
            num_envs, window_size, -1
        )
        actor_human_feature = np.concatenate(
            (
                human_anchor_rot6d,
                # human_joint_rot6d,
                ref_human_body_pos_in_ref_anchor,
                # ref_human_body_rot6d_in_ref_anchor,
            ),
            axis=-1,
        )
        actor_ref_human_fsq_feature_window = actor_human_feature.reshape(-1)
        return actor_ref_human_fsq_feature_window

    def sim_loop(self):
        for i in range(self.control_decimation):
            step_start = time.time()

            if not cfg.motion_play:
                # tau = self._PD_control()
                tau = self._PD_control(self.target_dof_pos)
                self.d.ctrl[:] = tau
            if not self.paused:
                self.prev_qpos = self.d.qpos.copy()
                self.set_camera()
                mujoco.mj_step(self.m, self.d)
            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _PD_control(self, _P_t=0):
        P_n = self.d.qpos[7:]
        V_n = self.d.qvel[6:]
        KP = self.P_gains
        KD = self.D_gains
        # 在_compute_torques中使用
        t = KP * (_P_t - P_n) - KD * V_n
        # t = KP * (_P_t - P_n) - KD * V_n
        return t

    def contact_force(self):
        force = 0
        for contact_id, contact in enumerate(self.d.contact):
            if contact.efc_address >= 0:  # Valid contact
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.d, contact_id, forcetorque)
                force += forcetorque[0]
        self.fz = force / 65 / 9.81

    def key_callback(self, keycode):
        # 按空格键切换暂停/继续

        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'running'}")
        elif chr(keycode).lower() == "w":
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0
            self.cmd[0] = 0.8
        elif chr(keycode).lower() == "s":
            self.cmd[0] = -0.8
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

        elif chr(keycode).lower() == "a":
            self.cmd[1] = 0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "d":
            self.cmd[1] = -0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "q":
            self.cmd[2] = 1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        elif chr(keycode).lower() == "e":
            self.cmd[2] = -1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        # 释放键时重置控制量
        elif keycode == 48:  # keycode=0 表示无按键
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

    def set_camera(self):
        # self.viewer.cam.distance = 4
        # self.viewer.cam.azimuth = 180  # 135
        # self.viewer.cam.elevation = 0.0
        # self.viewer.cam.fixedcamid = -1
        # self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # self.viewer.cam.trackbodyid = 1
        ...

    def init_vel_geom(self, input):
        # create an invisibale geom and add label on it
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
        self.vel_geom_id = self.viewer.user_scn.ngeom
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.array([0.2, 0.2, 0.2]),  # label_size
            pos=self.d.qpos[:3]
            + np.array(
                [0.0, 0.0, 1.0]
            ),  # lebel position, here is 1 meter above the root joint
            mat=np.eye(3).flatten(),  # label orientation, here is no rotation
            rgba=np.array([0, 0, 0, 0]),  # invisible
        )
        geom.label = str(input)  # set label text
        self.viewer.user_scn.ngeom += 1
        self._persistent_user_geom_count = self.viewer.user_scn.ngeom

    def update_vel_geom(self):
        # update the geom position and label text
        if self.vel_geom_id is None:
            return
        geom = self.viewer.user_scn.geoms[self.vel_geom_id]
        geom.pos = self.d.qpos[:3] + np.array([0.0, 0.0, 1.0])
        geom.label = "rb h{:.2f} \r\nGoal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z: {:.2f}".format(
            0.0,
            self.cmd[0],
            self.cmd[1],
            self.cmd[2],
            self.fz,
        )


if __name__ == "__main__":
    s = simulator()
    s.run()
