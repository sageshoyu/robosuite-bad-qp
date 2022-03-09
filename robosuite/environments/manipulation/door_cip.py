import numpy as np 

from robosuite.environments.manipulation.door import Door 
from robosuite.controllers import controller_factory
import robosuite.utils.transform_utils as T

import pybullet as p 

# Default DoorWithObstacles configuration
INITIAL_JOINTS = [3.186, 4.171, -0.019, 1.404, 1.544, 4.559, -4.437]
# INITIAL_JOINTS = [3.279,3.911,0.611,1.731,-0.435,4.195,-2.714]


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
    "num_obstacles": 2
}


class DoorCIP(Door):
    """docstring for DoorCIP"""
    def __init__(self,
                 robots,
                 env_configuration="default",
                 controller_configs=None,
                 gripper_types="default",
                 initialization_noise="default",
                 use_latch=False,
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
        self.num_obstacles = task_config["num_obstacles"]

        self.collisions = 0
        self.joint_limits = 0
        self.col_mags = []
        self.f_excess = 0
        self.t_excess = 0
        self.terminated = False
        self.early_termination = early_termination
        self.penalize_collisions = penalize_collisions

        self.myIK = None

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
        self.myIK = controller_factory("IK_POSE", ik_config)


    def _reset_internal(self):

        super()._reset_internal() 

        if self.ee_fixed_to_handle:

            self._setup_ik()

            ee_pos = self.robots[0].controller.ee_pos
            ee_ori_mat = self.robots[0].controller.ee_ori_mat
            ee_quat = T.mat2quat(ee_ori_mat)
            ee_in_world = T.pose2mat((ee_pos, ee_quat))
            
            site_pos = self.sim.data.site_xpos[self.door_handle_site_id] 
            site_ori_mat = self.sim.data.site_xmat[self.door_handle_site_id]
            site_ori_mat = np.array(site_ori_mat).reshape(3,3)

            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            door_quat = self.sim.model.body_quat[door_body_id]
            door_ori_mat = T.quat2mat(door_quat)
            target_ori_mat = ee_ori_mat 

            qpos = self.myIK.joint_pos_abs_ee(site_pos, target_ori_mat)

            self.sim.data.qpos[:7] = qpos
            self.robots[0].init_qpos = qpos
            self.robots[0].initialization_noise['magnitude'] = 0.0

        



