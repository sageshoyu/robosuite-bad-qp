import numpy as np 

from robosuite.environments.manipulation.door import Door 
from robosuite.controllers import controller_factory
import robosuite.utils.transform_utils as T

DEFAULT_DOOR_CONFIG = {
    # rewards and penalties
    "action_delta_penalty": 0,
    "arm_collision_penalty": -50,
    "door_shaped_reward": 30,
    "ee_accel_penalty": 0,
    "energy_penalty": 0,
    "excess_force_penalty_mul": 0.05,
    "excess_torque_penalty_mul": 5.0,
    "final_reward": 500,
    "handle_shaped_reward": 0.5,
    "handle_reward": True,
    "velocity_penalty": 10,

    # goals and thresholds
    "hinge_goal": 1.04,
    "dist_threshold": 0.01,
    "max_hinge_diff": 0.05,
    "max_hinge_vel": 0.1,
    "pressure_threshold_max": 100.0,
    "torque_threshold_max": 1.0,

    # file/env config
    "print_results": False,
    "ee_fixed_to_handle": True,
}


class DoorCIP(Door):
    """docstring for DoorCIP"""
    def __init__(self,
                 robots,
                 env_configuration="default",
                 controller_configs=None,
                 gripper_types="default",
                 initialization_noise="default",
                 use_latch=True,
                 use_camera_obs=True,
                 use_object_obs=True,
                 reward_scale=1.0,
                 reward_shaping=True,
                 early_termination=True,
                 penalize_collisions=True,
                 placement_initializer=None,  # TODO: figure out if this is EVER overidden
                 has_renderer=False,
                 has_offscreen_renderer=True,
                 render_camera="frontview",
                 render_collision_mesh=False,
                 render_visual_mesh=True,
                 render_gpu_device_id=-1,
                 control_freq=20,
                 horizon=1000,
                 ignore_done=False,
                 hard_reset=True,
                 camera_names="agentview",
                 camera_heights=256,
                 camera_widths=256,
                 camera_depths=False,
                 task_config=None):

        task_config = task_config if task_config is not None else DEFAULT_DOOR_CONFIG

        self.action_delta_penalty = task_config["action_delta_penalty"]
        self.arm_collision_penalty = task_config["arm_collision_penalty"]
        self.door_shaped_reward = task_config["door_shaped_reward"]
        self.ee_accel_penalty = task_config["ee_accel_penalty"]
        self.energy_penalty = task_config["energy_penalty"]
        self.excess_force_penalty_mul = task_config["excess_force_penalty_mul"]
        self.excess_torque_penalty_mul = task_config["excess_torque_penalty_mul"]
        self.final_reward = task_config["final_reward"]
        self.handle_reward = task_config["handle_reward"]
        self.velocity_penalty = task_config["velocity_penalty"]
        self.hinge_goal = task_config["hinge_goal"],
        self.max_hinge_diff = task_config["max_hinge_diff"]
        self.pressure_threshold_max = task_config["pressure_threshold_max"]
        self.torque_threshold_max = task_config["torque_threshold_max"]
        self.dist_threshold = task_config["dist_threshold"]
        self.handle_shaped_reward = task_config["handle_shaped_reward"]
        self.max_hinge_vel = task_config["max_hinge_vel"]
        self.print_results = task_config["print_results"]
        self.ee_fixed_to_handle = task_config["ee_fixed_to_handle"]

        self.collisions = 0
        self.joint_limits = 0
        self.col_mags = []
        self.f_excess = 0
        self.t_excess = 0
        self.terminated = False
        self.early_termination = early_termination
        self.penalize_collisions = penalize_collisions

        self.IK = None

        # super init 
        super().__init__(robots,
                         env_configuration,
                         controller_configs,
                         gripper_types,
                         initialization_noise,
                         use_latch,
                         use_camera_obs,
                         use_object_obs,
                         reward_scale,
                         reward_shaping,
                         placement_initializer,
                         has_renderer,
                         has_offscreen_renderer,
                         render_camera,
                         render_collision_mesh,
                         render_visual_mesh,
                         render_gpu_device_id,
                         control_freq,
                         horizon,
                         ignore_done,
                         hard_reset,
                         camera_names,
                         camera_heights,
                         camera_widths,
                         camera_depths)

    def _setup_ik(self):

        # IK solver 
        ik_config = self.robots[0].controller_config
        ik_config.pop("input_max", None)
        ik_config.pop("input_min", None)
        ik_config.pop("output_max", None)
        ik_config.pop("output_min", None)
        self.IK = controller_factory("IK_POSE", ik_config)


    def _reset_internal(self):

        super()._reset_internal()
        self.sim.forward()

        if self.ee_fixed_to_handle:

            self._setup_ik()

            # get ee pose
            ee_pos = self.robots[0].controller.ee_pos
            ee_ori_mat = self.robots[0].controller.ee_ori_mat
            ee_quat = T.mat2quat(ee_ori_mat)
            ee_in_world = T.pose2mat((ee_pos, ee_quat))
            
            # get pose of handle site, door 
            site_pos = self.sim.data.site_xpos[self.door_handle_site_id] 
            site_ori_mat = self.sim.data.site_xmat[self.door_handle_site_id]
            site_ori_mat = np.array(site_ori_mat).reshape(3,3)

            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            door_quat = self.sim.model.body_quat[door_body_id]
            door_ori_mat = T.quat2mat(door_quat)
            
            # compute target
            target_pos = site_pos
            R_x = T.rotation_matrix(-np.pi/2, np.array([1,0,0]))[:3,:3] 
            R_z = T.rotation_matrix(-np.pi/2, np.array([0,0,1]))[:3,:3]
            target_ori_mat = door_ori_mat @ R_x @ R_z 

            # ik 
            qpos = self.IK.ik(target_pos, target_ori_mat)

            # update sim 
            self.sim.data.qpos[:7] = qpos
            self.robots[0].init_qpos = qpos

    def reward(self, action=None):
        """
        Reward function for the task. 
        Modified from superclass so as to add dense reward on the hinge qpos as well. 

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the door is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between door handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by door handled
              - Note that this component is only relevant if the environment is using the locked door version

        Note that a successfully completed task (door opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward
            # Add rotating component if we're using a locked door
            if self.use_latch:
                handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
                reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

            # add hinge qpos component 
            hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
            reward += np.clip(hinge_qpos, 0, 0.5)

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # make sure the gripper stays closed
        # self.robots[0].grip_action(self.robots[0].gripper, [-1.0])

        # if terminating prematurely, signal episode end
        # if self._check_terminated():
        # if self.terminated:
        #     done = self.early_termination

        # # record collision and joint_limit info for logging
        # info["collisions"] = self.collisions
        # info["joint_limits"] = self.joint_limits
        # info['task_complete'] = self.sim.data.qpos[self.hinge_qpos_addr]
        # info["collision_forces"] = self.col_mags

        info["success"] = self._check_success()
        return reward, done, info