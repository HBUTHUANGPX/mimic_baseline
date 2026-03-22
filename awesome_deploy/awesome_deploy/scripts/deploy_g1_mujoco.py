from awesome_deploy.utils import VideoRecorder
from awesome_deploy.utils.math_func import *
from awesome_deploy.utils.cfg import cfg, current_path
from awesome_deploy.utils.infer import infere
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
        self.video_recorder = VideoRecorder(
            path=current_path + "/deploy_mujoco_recordings",
            tag=None,
            video_name="video_0",
            fps=int(1 / cfg.policy_dt),
            compress=False,
        )
        self.data_save = []

    def motion_play(self):
        t = int(self.time_step)
        self.d.qpos[0:3] = np.asarray(self.motion.body_pos_w[t, 7, :])
        self.d.qpos[0:2] = 0
        q = np.asarray(self.motion.body_quat_w[t, 0, :])
        self.d.qpos[3:7] = q
        self.d.qpos[7 : 7 + len(self.default_pos)] = np.asarray(
            self.motion.joint_pos[t]
        )[self.isaac_sim2mujoco_index]
        mujoco.mj_forward(self.m, self.d)
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
        self.renderer = mujoco.renderer.Renderer(self.m, height=480, width=640)
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
                break
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

    def policy_loop(self):
        # print("="*(20))
        self.counter += 1
        # print(self.d.qvel[0])
        self.qpos = self.d.qpos[7:]
        self.P_n = self.qpos - self.default_pos
        self.V_n = self.d.qvel[6:]

        # if self.time_step >= 100:
        if self.time_step >= self.motion.time_step_total:
            self.time_step = 10

        if cfg.motion_play:
            self.motion_play()
        else:
            self.minimum_infer()
        # print(f"time_step: {self.time_step}")
        self.contact_force()
        self.sim_loop()
        # 更新 Renderer 场景，使用查看器的相机和选项，使图像与窗口一致
        self.renderer.update_scene(
            self.d,
            camera=self.viewer.cam,  # 使用查看器的相机视图
            scene_option=self.viewer.opt,  # 使用查看器的渲染选项
        )

        # 捕获图像：返回 (height, width, 3) 的 uint8 NumPy 数组 (RGB)
        img = self.renderer.render()
        self.video_recorder(img)

        self.viewer.sync()
        self.update_vel_geom()

    def _obs_motion_joint_pos_command(self):
        return np.copy(self.motion.joint_pos[int(self.time_step)])

    def _obs_motion_joint_vel_command(self):
        return np.copy(self.motion.joint_vel[int(self.time_step)])

    def _obs_motion_ref_ori_b(self):
        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        _quat = self.pin.get_link_quaternion(cfg.motion_reference_body)
        self.robot_ref_quat_w = np.expand_dims(_quat, axis=0)  # shape [n,4]
        self.ref_quat_w = self.motion.body_quat_w[
            int(self.time_step),
            cfg.motion_body_names.index(cfg.motion_reference_body),
            :,
        ]  # shape [n,4]
        q01 = self.robot_ref_quat_w
        q02 = self.ref_quat_w
        if q02 is not None and q02.ndim == 1:
            q02 = np.expand_dims(q02, axis=0)
        q10 = quat_inv(q01)
        if q02 is not None:
            q12 = quat_mul(q10, q02)
        else:
            q12 = q10
        mat = matrix_from_quat(q12)
        motion_ref_ori_b = mat[..., :2].reshape(mat.shape[0], -1)  # shape [n,6]
        return motion_ref_ori_b

    def _obs_base_ang_vel(self):
        return self.d.qvel[3:6]

    def _obs_joint_pos(self):
        return (self.d.qpos[7:] - self.default_pos)[self.mujoco2isaac_sim_index]

    def _obs_joint_vel(self):
        return self.d.qvel[6:][self.mujoco2isaac_sim_index]

    def _obs_actions(self):
        return self.action

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

    def update_vel_geom(self):
        # update the geom position and label text
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1]
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
