"""Launches the MuJoCo sim-to-sim deployment loop for the active robot."""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from awesome_deploy.utils import VideoRecorder
from awesome_deploy.utils.cfg import cfg, current_path
from awesome_deploy.utils.infer import infere
from awesome_deploy.utils.math_func import matrix_from_quat, quat_inv, quat_mul

np.set_printoptions(precision=16, linewidth=100, threshold=np.inf, suppress=True)


class simulator(infere):
    """MuJoCo simulator that consumes actions from the deployment policy."""

    def __init__(self):
        """Loads the robot model, initializes policy inference, and allocates IO."""
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
        """Copies reference motion state directly into MuJoCo."""
        t = int(self.time_step)
        self.d.qpos[0:3] = np.asarray(self.motion.body_pos_w[t, 7, :])
        self.d.qpos[0:2] = 0
        q = np.asarray(self.motion.body_quat_w[t, 0, :])
        self.d.qpos[3:7] = q
        self.d.qpos[7 : 7 + len(self.default_pos)] = np.asarray(
            self.motion.joint_pos[t]
        )[self.isaac_sim2mujoco_index]
        mujoco.mj_forward(self.m, self.d)

    def _init_robot_conf(self):
        """Extends base robot config with MuJoCo body name lookup tables."""
        super()._init_robot_conf()
        self.mujoco_all_body_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.m.nbody)
        ][1:]
        self.mujoco_body_names_indices = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.mujoco_all_body_names
        ]
        self.motion_reference_body_index = cfg.motion_body_names.index(
            cfg.motion_reference_body
        )
        self._init_motion_window_offsets()
        print("mujoco_all_body_names:\r\n", self.mujoco_all_body_names)

    def _init_motion_window_offsets(self):
        """Loads the temporal window size used by the exported training setup."""
        self.command_window_offsets = np.asarray([0], dtype=np.int64)
        env_cfg_path = os.path.join(cfg.policy_dir, "env.yaml")
        if not os.path.isfile(env_cfg_path):
            return
        with open(env_cfg_path, "r", encoding="utf-8") as file:
            env_cfg = yaml.safe_load(file) or {}
        motion_cfg = env_cfg.get("commands", {}).get("motion", {})
        history_frames = int(motion_cfg.get("history_frames", 0))
        future_frames = int(motion_cfg.get("future_frames", 0))
        print("history_frames: ",history_frames)
        print("future_frames: ",future_frames)
        self.command_window_offsets = np.arange(
            -history_frames,
            future_frames + 1,
            dtype=np.int64,
        )

    def run(self):
        """Runs the full rollout loop until the viewer closes or motion ends."""
        save_data_flag = 1
        self.counter = 0
        self.target_dof_pos = self.default_pos.copy()[: self.action_num]
        self.phase = 0
        if save_data_flag:
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
                # Align the simulator with the first motion frame before the
                # closed-loop rollout starts.
                if cfg.motion_play:
                    self.motion_play()
                    self.time_step = 0
                else:
                    self.motion_play()
                mujoco.mj_step(self.m, self.d)
                self.viewer.sync()

            self.policy_loop()
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
            if self.time_step >= self.motion.time_step_total:
                break

        for key in (
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
            log[key] = np.stack(log[key], axis=0)
        save_path = current_path + "/tmp/motion.npz"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **log)
        print("stop")
        self.video_recorder.stop()

    def policy_loop(self):
        """Runs one control update including inference, logging, and rendering."""
        self.counter += 1
        self.qpos = self.d.qpos[7:]
        self.P_n = self.qpos - self.default_pos
        self.V_n = self.d.qvel[6:]

        if self.time_step >= self.motion.time_step_total:
            self.time_step = 10

        if cfg.motion_play:
            self.motion_play()
        else:
            self.minimum_infer()

        self.contact_force()
        self.sim_loop()
        # Render from the same camera/view options shown in the interactive
        # viewer so the saved video matches what the operator sees.
        self.renderer.update_scene(
            self.d,
            camera=self.viewer.cam,
            scene_option=self.viewer.opt,
        )
        img = self.renderer.render()
        self.video_recorder(img)
        self.viewer.sync()
        self.update_vel_geom()

    def _obs_motion_joint_pos_command(self):
        """Returns the current reference motion joint positions."""
        return np.copy(self.motion.joint_pos[int(self.time_step)])

    def _obs_motion_joint_vel_command(self):
        """Returns the current reference motion joint velocities."""
        return np.copy(self.motion.joint_vel[int(self.time_step)])

    def _obs_joint_pos_delta(self):

        return np.copy(self.motion.joint_pos[int(self.time_step)])-self._obs_joint_pos()
    
    def _obs_robot_joint_pos(self):

        return np.copy(self.motion.joint_pos[int(self.time_step)])

    def _obs_motion_ref_ori_b(self):
        """Returns body-frame orientation error to the motion reference."""
        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        ref_body_index = cfg.motion_body_names.index(cfg.motion_reference_body)
        self.robot_ref_quat_w = np.expand_dims(
            self.pin.get_link_quaternion(cfg.motion_reference_body), 
            axis=0
        )
        self.ref_quat_w = self.motion.body_quat_w[
            int(self.time_step), ref_body_index, :
        ]
        q01 = self.robot_ref_quat_w
        q02 = self.ref_quat_w
        if q02 is not None and q02.ndim == 1:
            q02 = np.expand_dims(q02, axis=0)
        # Convert the world-frame orientation delta into a compact body-frame
        # representation used by the current observation design.
        q10 = quat_inv(q01)
        if q02 is not None:
            q12 = quat_mul(q10, q02)
        else:
            q12 = q10
        mat = matrix_from_quat(q12)
        return mat[..., :2].reshape(mat.shape[0], -1)

    def _obs_base_ang_vel(self):
        """Returns the MuJoCo base angular velocity."""
        return self.d.qvel[3:6]

    def _obs_joint_pos(self):
        """Returns joint positions in policy joint order."""
        return (self.d.qpos[7:] - self.default_pos)[self.mujoco2isaac_sim_index]

    def _obs_joint_vel(self):
        """Returns joint velocities in policy joint order."""
        return self.d.qvel[6:][self.mujoco2isaac_sim_index]

    def _obs_actions(self):
        """Returns the most recent policy action vector."""
        return self.action

    def _get_motion_window_indices(self):
        """Returns clipped motion indices for the active temporal window."""
        window_indices = int(self.time_step) + self.command_window_offsets
        return np.clip(window_indices, 0, self.motion.time_step_total - 1)

    def _obs_joint_pos_delta_window(self):
        """Returns flattened joint-position deltas for the temporal window."""
        motion_joint_pos_window = np.copy(
            self.motion.joint_pos[self._get_motion_window_indices()]
        )
        joint_pos = np.expand_dims(self._obs_joint_pos(), axis=0)
        return (motion_joint_pos_window - joint_pos).reshape(-1)

    def _obs_robot_joint_pos_window(self):
        """Returns flattened target joint positions for the temporal window."""
        return np.copy(
            self.motion.joint_pos[self._get_motion_window_indices()]
        ).reshape(-1)

    def _obs_motion_ref_ori_b_window(self):
        """Returns flattened 6D reference orientations for the temporal window."""
        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        robot_ref_quat_w = np.expand_dims(
            self.pin.get_link_quaternion(cfg.motion_reference_body), axis=0
        )
        motion_ref_quat_w = np.copy(
            self.motion.body_quat_w[
                self._get_motion_window_indices(),
                self.motion_reference_body_index,
                :,
            ]
        )
        robot_ref_quat_w = np.repeat(robot_ref_quat_w, len(motion_ref_quat_w), axis=0)
        rel_quat_b = quat_mul(quat_inv(robot_ref_quat_w), motion_ref_quat_w)
        rel_mat_b = matrix_from_quat(rel_quat_b)
        return rel_mat_b[..., :2].reshape(-1)

    def sim_loop(self):
        """Advances MuJoCo for one policy interval using PD control."""
        for _ in range(self.control_decimation):
            step_start = time.time()
            if not cfg.motion_play:
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
        """Computes actuator torques from target joint positions.

        Args:
            _P_t: Desired joint positions in MuJoCo joint order.

        Returns:
            Torque command vector for MuJoCo actuators.
        """
        P_n = self.d.qpos[7:]
        V_n = self.d.qvel[6:]
        KP = self.P_gains
        KD = self.D_gains
        return KP * (_P_t - P_n) - KD * V_n

    def contact_force(self):
        """Accumulates vertical contact force and normalizes it by body weight."""
        force = 0
        for contact_id, contact in enumerate(self.d.contact):
            if contact.efc_address >= 0:
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.d, contact_id, forcetorque)
                force += forcetorque[0]
        self.fz = force / 65 / 9.81

    def key_callback(self, keycode):
        """Handles keyboard commands for pause and commanded velocity changes."""
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
        elif keycode == 48:
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

    def set_camera(self):
        """Updates the viewer camera if a custom camera policy is desired."""
        ...

    def init_vel_geom(self, input):
        """Creates an on-screen MuJoCo label for command and force feedback.

        Args:
            input: Initial label text.
        """
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.array([0.2, 0.2, 0.2]),
            pos=self.d.qpos[:3] + np.array([0.0, 0.0, 1.0]),
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0, 0, 0]),
        )
        geom.label = str(input)
        self.viewer.user_scn.ngeom += 1

    def update_vel_geom(self):
        """Refreshes the runtime status label shown above the robot."""
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1]
        geom.pos = self.d.qpos[:3] + np.array([0.0, 0.0, 1.0])
        geom.label = (
            "rb h{:.2f} \r\nGoal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z: {:.2f}"
        ).format(
            0.0,
            self.cmd[0],
            self.cmd[1],
            self.cmd[2],
            self.fz,
        )


if __name__ == "__main__":
    s = simulator()
    s.run()
