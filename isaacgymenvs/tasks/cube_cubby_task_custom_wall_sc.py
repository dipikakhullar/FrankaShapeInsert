import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
import torch
# from utils import *
offset = 9

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat




class CubeCubbyTaskCustomWallSC(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.walls = []

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
            "r_quat_scale": self.cfg["env"]["quatRewardScale"],
            "r_quat_scale_2": self.cfg["env"]["quatRewardScale2"],
            "rot_eps": self.cfg["env"]["rotEps"]
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeA_to_cubby (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 26
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cylinder_state = None        # Initial state of cylinder for current env
        # self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cylinder_state = None
        # self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._wall_id= None                   # Actor ID corresponding to MUG for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.global_step_counter = 0  # Initialize step counter


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)


        self.success_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Per-episode counter
        # self.total_success_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Tracks across episodes


        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        if self.viewer != None:
            self._set_viewer_params()


    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-2.0, -2.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        # Look at the first env
	# cam_pos = gymapi.Vec3(3, 1.5, 10)
	# cam_target = gymapi.Vec3(0, 0.0, 0)
	# self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def create_cubby(self, env_ptr, cubby_center, frame_bar_length, frame_bar_thickness, num_frames, frame_spacing=0):
        """
        Create a cubby made of stacked square frames and add it as a single aggregated actor to the environment.
        
        Args:
        - env_ptr: Environment pointer to which the cubby should be added.
        - cubby_center: Center position of the cubby in the environment.
        - frame_bar_length: Length of the bars forming the frame.
        - frame_bar_thickness: Thickness of the bars forming the frame.
        - num_frames: Number of frames to stack.
        - frame_spacing: Vertical spacing between frames.
        """
        
        # Create bar assets (horizontal and vertical)
        bar_options = gymapi.AssetOptions()
        bar_options.fix_base_link = True
        
        horizontal_bar = self.gym.create_box(self.sim, frame_bar_length, frame_bar_thickness, frame_bar_thickness, bar_options)
        vertical_bar = self.gym.create_box(self.sim, frame_bar_thickness, frame_bar_length + frame_bar_thickness, frame_bar_thickness, bar_options)

        # Maximum number of aggregated bodies and shapes for the cubby
        max_agg_bodies = num_frames * 4  # 4 bars per frame
        max_agg_shapes = num_frames * 4

        # Begin aggregation (group the bars into one actor)
        # self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

        cubby_rotation0 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 0), 1.5708)  # Rotate 90 degrees about Y-axis
        cubby_rotation1 = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.5708)  # Rotate 90 degrees about Y-axis
        cubby_rotation2 = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.5708)  # Rotate 90 degrees about Y-axis
        self._cubby_top_ids = [None]*num_frames
        self._cubby_bottom_ids = [None]*num_frames
        self._cubby_left_ids = [None]*num_frames
        self._cubby_right_ids = [None]*num_frames
        # Loop through each frame to create the cubby
        for j in range(num_frames):
            # Top horizontal bar
            frame_start_pose = gymapi.Transform()
            frame_start_pose.p = gymapi.Vec3(cubby_center[0], cubby_center[1], cubby_center[2] + frame_bar_length / 2)
            frame_start_pose.r = cubby_rotation1
            self._cubby_top_ids[j] = self.gym.create_actor(env_ptr, horizontal_bar, frame_start_pose, f"top_bar_{j}", -1, 0)

            # Bottom horizontal bar
            frame_start_pose.p = gymapi.Vec3(cubby_center[0], cubby_center[1], cubby_center[2] - frame_bar_length / 2)
            frame_start_pose.r = cubby_rotation1  # Apply the cubby rotation
            self._cubby_bottom_ids[j]  =self.gym.create_actor(env_ptr, horizontal_bar, frame_start_pose, f"bottom_bar_{j}", -1, 0)

            # # # Left vertical bar
            # frame_start_pose.p = gymapi.Vec3(cubby_center[0] - frame_bar_length / 2, cubby_center[1], z_offset)
            frame_start_pose.p = gymapi.Vec3(cubby_center[0], cubby_center[1] + frame_bar_length / 2, cubby_center[2])
            frame_start_pose.r = cubby_rotation2  # Apply the global cubby rotation

            self._cubby_left_ids[j] =self.gym.create_actor(env_ptr, vertical_bar, frame_start_pose, f"left_bar_{j}", -1, 0)

            # # # Right vertical bar
            frame_start_pose.p = gymapi.Vec3(cubby_center[0], cubby_center[1] - frame_bar_length / 2, cubby_center[2])
            frame_start_pose.r = cubby_rotation2  # Apply the global cubby rotation

            self._cubby_right_ids[j] = self.gym.create_actor(env_ptr, vertical_bar, frame_start_pose, f"right_bar_{j}", -1, 0)

        # End aggregation to treat the cubby as one actor
        # self.gym.end_aggregate(env_ptr)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        wall_asset_file =  "urdf/separated_wall_shapes/square_bottom.urdf" #"urdf/shape_wall_two_holes/shape_wall_two_holes_higher.urdf"
        cylinder_asset_file = "urdf/cylinder.urdf"

        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)


        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        

        wall_opts = gymapi.AssetOptions()
        wall_opts.fix_base_link = True
        wall_opts.use_mesh_materials = True  
        wall_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
        wall_opts.override_com = True 
        wall_opts.override_inertia = True 
        wall_opts.disable_gravity = True
        wall_opts.vhacd_enabled = True 
        wall_opts.vhacd_params = gymapi.VhacdParams() 
        wall_opts.vhacd_params.resolution = 3000
        wall_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        wall_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_file, wall_opts)


        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cubeA_size = 0.050
        # self.cubeB_size = 0.070

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        self.cylinder_radius = 0.05
        self.cylinder_length = 0.06
        cylinder_opts = gymapi.AssetOptions()
        cylinder_asset = self.gym.load_asset(self.sim, asset_root, cylinder_asset_file, asset_options)
        cylinder_color = gymapi.Vec3(0.0, 0.4, 0.1)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        cylinder_start_pose = gymapi.Transform()
        cylinder_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cylinder_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # wall stuff 
        self.table_height = 1
        self.wall_height = 1.0
        self.square_center_height = .3
        self.square_length = .3
        self.circle_center_height = .7
        self.circle_diameter = .3
        self.wall_center_to_square_top = .05
        self.wall_thickness = .25

        # is the table height midle of thickness of table or top of thickness? 
        # assume it's center for now, verify later TO DO
        self.ground_to_square_bottom = self.table_height + table_thickness/2 + .15
        self.ground_to_square_bottom_tensor = torch.tensor([0,0,self.table_height + table_thickness/2 + .15]).repeat(num_envs, 1)
        # print("self.ground_to_square_bottom_tensor: ", self.ground_to_square_bottom_tensor.shape)
        self.ground_to_circle_bottom = self.table_height + table_thickness/2 + .55
        self.ground_to_circle_bottom_tensor = torch.tensor([0,0,self.table_height + table_thickness/2 + .55])
        self.wall_pos = [0.3, 0, self.table_height + table_thickness + self.wall_height/2]
        # self.circle_hole_center = [0, 0, self.table_height + table_thickness + self.square_center_height]
        # self.square_hole_center = [0, 0, self.table_height + table_thickness + self.circle_center_height]
        wall_start_pose = gymapi.Transform()
        wall_start_pose.p = gymapi.Vec3(*self.wall_pos)
        wall_rot = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -1.5708)  # Rotate 90 degrees about Y-axis

        wall_start_pose.r = wall_rot

        # print("wall_start_pose.p: ", self.wall_pos)
        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + offset    # 1 for table, table stand, cubeA, cubeB, wall, wall, + 4 boxes for frames
        max_agg_shapes = num_franka_shapes + offset    # 1 for table, table stand, cubeA, cubeB, wall, wall

        self.frankas = []
        self.envs = []


        # # Define cubby parameters
        # self.frame_bar_length = 0.5  # Length of the bars forming the frame
        # self.frame_bar_thickness = .2  # Thickness of the bars forming the frame
        # num_frames = 1  # Number of frames to stack
        # frame_spacing = 0.3  # Vertical spacing between frames
        # table_height = 1
        
        # print("z height for cubby; ", table_height + table_thickness/2  + self.frame_bar_length/2)
        # cubby_center = [.3, 0, table_height + table_thickness + self.frame_bar_length/2]  # Cubby position
        # cubby_center_tensor = torch.tensor(cubby_center, device=self.device, dtype=torch.long)

        # cubby_bottom = torch.tensor([cubby_center[0], cubby_center[1], cubby_center[2] - self.frame_bar_length/2 + self.frame_bar_thickness/2],  device=self.device, dtype=torch.long)
        # self.target_cube_center = torch.tensor([cubby_center[0], cubby_center[1], cubby_center[2] - self.frame_bar_length/2 + self.frame_bar_thickness/2 + self.cubeA_size],  device=self.device, dtype=torch.long)
        for i in range(self.num_envs):
            # create env instance

            # print(self.aggregate_mode)
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            
            # 1. Franka
            # print("franka", self.aggregate_mode)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 2. Custom Wall
            # print("create wall actor")
            self._wall_id = self.gym.create_actor(env_ptr, wall_asset, wall_start_pose, "wall", i, 0, 0)

            # 3. Table
            # print("table", self.aggregate_mode)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            # 4. Cube A
            # print("cubeA", self.aggregate_mode)
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)

            # 5. Cylinder B
            # print("cubeB", self.aggregate_mode)
            self._cylinder_id = self.gym.create_actor(env_ptr, cylinder_asset, cylinder_start_pose, "cylinder", i, 4, 0)
            self.gym.set_rigid_body_color(env_ptr, self._cylinder_id, 0, gymapi.MESH_VISUAL, cylinder_color)


            # print_collision_info(self.gym, self.sim, self._cubeA_id, env_ptr, "cubeA")
            # print_collision_info(self.gym, self.sim, self._wall_id, env_ptr, "wall")

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            # print("SELF.ENVS", self.envs)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cylinder_state = torch.zeros(self.num_envs, 13, device=self.device)
        # self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)
        cube_goal_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.5708)
        self._desired_cube_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_cube_quat[:, 0] = cube_goal_quat.x
        self._desired_cube_quat[:, 1] = cube_goal_quat.y
        self._desired_cube_quat[:, 2] = cube_goal_quat.z
        self._desired_cube_quat[:, 3] = cube_goal_quat.w
        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            # "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
            # Cubbies
            #at some point we need to not havea  list of num frames bc we aren't stacking frames anymore we are just increasing the thickness of the box.
            # "cubby_bottom_handle_{}".format(0): self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubby_bottom_ids[0], "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cylinder_state = self._root_state[:, self._cylinder_id, :]
        self._wall_state = self._root_state[:, self._wall_id, :]

        # 
        # self._cubby_bottom_state = self._root_state[:, self._cubby_bottom_ids[0], :]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cylinder_length": torch.ones_like(self._eef_state[:, 0]) * self.cylinder_length,
            "cylinder_radius": torch.ones_like(self._eef_state[:, 0]) * self.cylinder_radius,
            "target_quat": self._desired_cube_quat
            # "frame_bar_thickness": torch.ones_like(self._eef_state[:, 0]) * self.frame_bar_thickness,
            # "frame_bar_length": torch.ones_like(self._eef_state[:, 0]) * self.frame_bar_length
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * offset, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)


        # Print out global indices for the first environment to verify actor order
        # print("Global indices for environment 0:", self._global_indices[0])
        
    def _update_states(self):
        self.states.update({
            # Franka Robot
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            "eef_state": self._eef_state[:, :3],

            # Cube A 
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3], # cubeA position relative to hand. How far is cube A from hand.
            "cubeA_to_wall_center": self._cubeA_state[:, :3] - self._wall_state[:, :3],
            "cubeA_vel": self._rigid_body_state[:, self._cubeA_id, 7:10],  # Linear velocity (assuming index 7:10 is linear velocity)
            "cubeA_ang_vel": self._rigid_body_state[:, self._cubeA_id, 10:13],  # Angular velocity (assuming index 10:13 is angular velocity),
            "cube_A_to_square_bottom": self._cubeA_state[:, :3] - self.ground_to_square_bottom,
            
            # Cylinder
            "cylinder_length": torch.ones_like(self._eef_state[:, 0]) * self.cylinder_length,
            "cylinder_radius": torch.ones_like(self._eef_state[:, 0]) * self.cylinder_radius,
            "cylinder_quat": self._cylinder_state[:, 3:7],
            "cylinder_pos": self._cylinder_state[:, :3],
            "cylinder_pos_relative": self._cylinder_state[:, :3] - self._eef_state[:, :3], # cylinder position relative to hand. How far is cube A from hand.
            "cylinder_to_wall_center": self._cylinder_state[:, :3] - self._wall_state[:, :3],
            "cylinder_vel": self._rigid_body_state[:, self._cylinder_id, 7:10],  # Linear velocity (assuming index 7:10 is linear velocity)
            "cylinder_ang_vel": self._rigid_body_state[:, self._cylinder_id, 10:13],  # Angular velocity (assuming index 10:13 is angular velocity),
            "cylinder_to_circle_bottom": self._cylinder_state[:, :3] - self.ground_to_circle_bottom,

            # Wall
            "wall_pos": self._wall_state[:, :3], # wall center
            "wall_center_to_square_top": torch.ones_like(self._eef_state[:, 2]) * self.wall_center_to_square_top,
            "square_length": torch.ones_like(self._eef_state[:, 2]) * self.square_length,

            # Evaluation
            "success_counter": self.success_counter,
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        # obs include: cubeA_pose (7) + cubeA_to_cube_A_to_square_bottom (3) + eef_pose (7) + q_gripper (2)
        # idea: generalize this to obs = [obj_quat, obj_pos, obj_to_goal, eef_pos, eef_quat]
        obs = ["cubeA_quat", "cubeA_pos", "cube_A_to_square_bottom", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        # print("FUNC: RESET_IDX()")
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        # self._reset_init_object_state(cube='B', env_ids=env_ids, check_valid=False)
        self._reset_init_object_state(obj='cube', env_ids=env_ids, check_valid=True)
        self._reset_init_object_state(obj='cylinder', env_ids=env_ids, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cylinder_state[env_ids] = self._init_cylinder_state[env_ids]
        # self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
    
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_object_state(self, obj, env_ids, check_valid=True):
        """
        Simple method to reset @obj's position based on fixed initial setup.
        Populates the appropriate self._init_objX_state for obj A.

        Args:
            obj(str): Which obj to sample location for. Either 'cube' or 'cylinder'
            env_ids (tensor or None): Specific environments to reset obj for
            check_valid (bool): Not used anymore since position is fixed.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_obj_state = torch.zeros(num_resets, 13, device=self.device)
        y_offsets = torch.zeros(num_resets, device=self.device)

        # Get correct references depending on which one was selected
        if obj.lower() == 'cube':
            this_obj_state_all = self._init_cubeA_state
            obj_heights = self.states["cubeA_size"]
            y_offsets = torch.ones(num_resets, device=self.device)*.05
        elif obj.lower() == 'cylinder':
            this_obj_state_all = self._init_cylinder_state
            obj_heights = self.states["cylinder_length"]
            y_offsets = torch.ones(num_resets, device=self.device)*.05*-1
        else:
            raise ValueError(f"Invalid cube specified, options are 'A'; got: {cube}")

        # Define the fixed position for cube A, centered on the table
        centered_obj_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set fixed X and Y values for obj A (centered position)
        sampled_obj_state[:, :2] = centered_obj_xy_state.unsqueeze(0)

        # change y position of object so they aren't stacked
        sampled_obj_state[:, 1] = sampled_obj_state[:, 1] + y_offsets.unsqueeze(0)

        # Set the Z value (fixed height)
        sampled_obj_state[:, 2] = self._table_surface_pos[2] + obj_heights[env_ids] / 2

        # Initialize rotation with no rotation (quat w = 1)
        sampled_obj_state[:, 6] = 1.0

        # If you still want to add rotational noise, keep this part
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_obj_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_obj_state[:, 3:7])

        # Set the new initial state for obj A
        this_obj_state_all[env_ids, :] = sampled_obj_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        # Increment step counter
        # self.progress_buf is a tensor with size num_envs, where each element corresponds to the number of steps for the respective environment.
        # self.global_step_counter is a single value that tracks the total number of steps across all environments globally.
        # print("Steps in this episode for each environment:", self.progress_buf[0].item(), "global step counter: ", self.global_step_counter)
        self.global_step_counter += 1
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        # print reward here

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            # cubeB_pos = self.states["cubeB_pos"]
            # cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):  

    # TODO add rotation flag (if quat_diff_norm < value)

    # Define the cylinder-related variables from states
    cylinder_length = states["cylinder_length"]  # Length of the cylinder
    cylinder_radius = states["cylinder_radius"]  # Radius of the cylinder
    cylinder_quat = states["cylinder_quat"]  # Quaternion for the cylinder orientation
    cylinder_pos = states["cylinder_pos"]  # Cylinder position (x, y, z)
    cylinder_pos_relative = states["cylinder_pos_relative"]  # Cylinder position relative to the end effector (EEF)
    cylinder_to_wall_center = states["cylinder_to_wall_center"]  # Distance from the cylinder to the wall center
    cylinder_vel = states["cylinder_vel"]  # Linear velocity of the cylinder
    cylinder_ang_vel = states["cylinder_ang_vel"]  # Angular velocity of the cylinder
    cylinder_to_circle_bottom = states["cylinder_to_circle_bottom"]  # Distance from the cylinder to the circle bottom

    # Compute per-env physical parameters
    # TO DO why aren't we using wlal thicknes anywhere?? WE DON'T WANT HADN STUCK IN HOLE
    cubeA_size = states["cubeA_size"] #shape = (num_envs)
    square_length = states["square_length"]
    wall_pos = states["wall_pos"]
    wall_center_to_square_top = states["wall_center_to_square_top"]
    # z value of wall center - wall_center_to_square_top - square_length + cubeA_size/2 --> z value of center of where cubeA is supposed to be
    target_height = wall_pos[:, 2] - wall_center_to_square_top - square_length + cubeA_size/2

    # distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # reward for lifting cubeA
    cubeA_height_from_table = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
    cubeA_height_from_ground = states["cubeA_pos"][:, 2]
    cubeA_lifted = (cubeA_height_from_table - cubeA_size) > 0.04
    lift_reward = cubeA_lifted

    # my method
    #rotation reward
    # Orientation alignment for the cube in hand and goal cube
    # target_quat = states['target_quat']
    # quat_diff = quat_mul(states['cubeA_quat'], quat_conjugate(target_quat))
    # quat_diff_norm = torch.norm(quat_diff, dim=-1)
    # quat_align_reward = (1 - torch.tanh(10*quat_diff_norm))* cubeA_lifted


    # method from shadow_hand.py
    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(states['cubeA_quat'], quat_conjugate(target_quat))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    # quat_align_reward_2 = 1.0/(torch.abs(rot_dist) + reward_settings["rot_eps"])* cubeA_lifted
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]
    # print(f"FUNC: compute_franka_reward(): wall_pos: [{states['wall_pos'][0, 0].item()}, {states['wall_pos'][0, 1].item()}, {states['wall_pos'][0, 2].item()}]")

    # offset = center of cube in success position with respect to center of wall
    # z_offset: wall_pos[2] - wall_center_to_square_top[2] - square_length[2] + cubeA_size/2
    # how closely aligned cubeA is to cubby (only provided if cubeA is lifted) num_envs X 3 
    offset = torch.zeros_like(states["cubeA_to_cubby"])
    offset[:, 2] = wall_center_to_square_top + cubeA_size/2
    d_a_square_bottom = torch.norm(states["cube_A_to_square_bottom"] + offset, dim=-1)
    align_reward = (1 - torch.tanh(10.0 * d_a_square_bottom)) * cubeA_lifted
    # Dist reward is maximum of dist and align reward
    dist_reward_max = torch.max(dist_reward, align_reward)

    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    # x,y values of cubeA where it's supposed to sit x,y. Dist(current cubeA xy, goal x,y)
    xy_cubeA_to_square_hole = states["cubeA_to_wall_center"][:, :2]
    cubeA_align_square_hole = (torch.norm(xy_cubeA_to_square_hole, dim=-1) < 0.02)

    # cubeA_on_square_hole reward is relative to the table and cubeA_height is relative to table
    cubeA_on_square_hole = torch.abs(cubeA_height_from_ground - target_height) < 0.02
    gripper_away_from_cubeA = (d > 0.04)
    # stack reward is a boolean condition that tells us if we have succeeded
    stack_reward = cubeA_align_square_hole & cubeA_on_square_hole & gripper_away_from_cubeA
    # Compose rewards


    # ---- Cylinder-related rewards and sub-goals ---- #    
    # 1. Compute the alignment of the cylinder to the cylindrical hole
    cylinder_align_reward = (1 - torch.tanh(10.0 * torch.norm(cylinder_to_circle_bottom, dim=-1)))

    # 2. Orientation reward for the cylinder
    # desired_cylinder_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=cylinder_quat.device)
    # cylinder_quat_diff = quat_mul(cylinder_quat, quat_conjugate(desired_cylinder_orientation))
    # cylinder_rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(cylinder_quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    # cylinder_orientation_reward = 1.0 / (torch.abs(cylinder_rot_dist) + reward_settings["rot_eps"])

    # 3. Check if the cylinder is aligned with the cylindrical hole
    xy_cylinder_to_hole = torch.norm(cylinder_to_wall_center[:, :2], dim=-1)
    cylinder_align_hole = (xy_cylinder_to_hole < 0.02)  # Boolean for XY alignment

    # 4. Check if the cylinder is at the correct height (aligned with the bottom of the cylindrical hole)
    cylinder_on_circle_hole = torch.abs(cylinder_to_circle_bottom[:, 2]) < 0.02  # Boolean for height alignment

    # 5. Check if the gripper is away from the cylinder
    gripper_away_from_cylinder = torch.norm(cylinder_pos_relative, dim=-1) > 0.04  # Boolean for gripper away

    # 6. Final cylinder placement reward (success if all sub-goals are met)
    cylinder_stack_reward = cylinder_align_hole & cylinder_on_circle_hole & gripper_away_from_cylinder

    # # ---- Combine rewards ---- #
    # cylinder_reward = reward_settings["r_dist_scale"] * cylinder_align_reward + \
    #                   reward_settings["r_quat_scale"] * cylinder_orientation_reward


    # We either provide the stack reward or the align + dist reward
    # if stack_rewared == True, return r_stack_scale*(1), else return r_dist_scale*dist_reward + r_lift_scale*lift_reward + r_align_scale*align_reward
    rewards = torch.where(
        # if stackrewrd is true
        stack_reward,
        #16
        reward_settings["r_stack_scale"] * stack_reward,
        # otherwise if stack reward is false
        reward_settings["r_dist_scale"] * dist_reward_max + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
            "r_align_scale"] * align_reward #+ reward_settings["r_quat_scale"]*quat_align_reward + reward_settings["r_quat_scale_2"]*quat_align_reward_2,
    )


    # # Update the success count for each environment where the stack_reward is True
    for i in range(stack_reward.shape[0]):
        if stack_reward[i].item():
            states["success_counter"][i] += 1

    # # Print success information after a certain number of steps or episodes
    # if progress_buf[0].item() % 1000 == 0:  # Adjust this frequency as needed
    #     total_successes = success_counter.sum().item()
    #     print(f"Total Successes: {total_successes} across all environments")
    #     print(f"Successes per environment: {success_counter}")


    # Print which conditions are true for which environment
    # does this need to be chagned? or gripper_away_from_cubeA[i].item() TO DO
    for i in range(stack_reward.shape[0]):
        if cubeA_align_square_hole[i].item() or cubeA_on_square_hole[i].item() or cubeA_lifted[i].item():
            print(f"Env {i}:")
            print(f"  cubeA_lifted:         {cubeA_lifted[i].item()}")
            print(f"  cubeA_align_square_hole: {cubeA_align_square_hole[i].item()}")
            print("cubeA align xy expected: ", [states['wall_pos'][i, 0].item(), states['wall_pos'][i, 1].item()], "xy actual:" , states['cubeA_pos'][i, 0].item(), states['cubeA_pos'][i, 1].item())
            print(f"  cubeA_on_square_hole:    {cubeA_on_square_hole[i].item()} (cubeA_height_from_ground: {cubeA_height_from_ground[i].item()} vs target_height: {target_height[i].item()})")
            print(f"  gripper_away_from_cubeA: {gripper_away_from_cubeA[i].item()}")
            
            # Print positions
            print(f"  cubeA_size:            {cubeA_size[i].item()}")
            print(f"  cubeA_pos:             [{states['cubeA_pos'][i, 0].item()}, {states['cubeA_pos'][i, 1].item()}, {states['cubeA_pos'][i, 2].item()}]")
            print(f"  eef_lf_pos:            [{states['eef_lf_pos'][i, 0].item()}, {states['eef_lf_pos'][i, 1].item()}, {states['eef_lf_pos'][i, 2].item()}]")
            print(f"  eef_rf_pos:            [{states['eef_rf_pos'][i, 0].item()}, {states['eef_rf_pos'][i, 1].item()}, {states['eef_rf_pos'][i, 2].item()}]")
            print(f"  eef_state:             [{states['eef_state'][i, 0].item()}, {states['eef_state'][i, 1].item()}, {states['eef_state'][i, 2].item()}]")
            print(f"  cubeA_pos_relative:    [{states['cubeA_pos_relative'][i, 0].item()}, {states['cubeA_pos_relative'][i, 1].item()}, {states['cubeA_pos_relative'][i, 2].item()}]")
            print(f"  cylinder_pos:          [{states['cylinder_pos'][i, 0].item()}, {states['cylinder_pos'][i, 1].item()}, {states['cylinder_pos'][i, 2].item()}]")
            # If you want to print the wall position, add this:
            print(f"  wall_pos:              [{states['wall_pos'][i, 0].item()}, {states['wall_pos'][i, 1].item()}, {states['wall_pos'][i, 2].item()}]")
            print(f"  d_a_square_bottom:      {d_a_square_bottom[i]}")
            # Print velocities
            print(f"  cubeA_linear_velocity: [{states['cubeA_vel'][i, 0].item()}, {states['cubeA_vel'][i, 1].item()}, {states['cubeA_vel'][i, 2].item()}]")
            print(f"  cubeA_angular_velocity:[{states['cubeA_ang_vel'][i, 0].item()}, {states['cubeA_ang_vel'][i, 1].item()}, {states['cubeA_ang_vel'][i, 2].item()}]")

    # # Update the success count for each environment where the stack_reward is True
    # for i in range(stack_reward.shape[0]):
    #     if stack_reward[i].item():
    #         success_counter[i] += 1

    # # Print success information after a certain number of steps or episodes
    # if progress_buf[0].item() % 1000 == 0:  # Adjust this frequency as needed
    #     total_successes = success_counter.sum().item()
    #     print(f"Total Successes: {total_successes} across all environments")
    #     print(f"Successes per environment: {success_counter}")

    # if progress_buf[0].item() == (max_episode_length-1):
    #     for i in range(1):
    #         # [progress_buff, ]
    #         print_str = ""
    #         print_str += "id: {}, ".format(i)
    #         print_str += "{}, ".format(progress_buf[i].item())
    #         print_str += "{}, ".format(dist_reward[i].item())
    #         print_str += "{}, ".format(cubeA_lifted[i].item())
    #         print_str += "{}, ".format(rot_dist[i].item())
    #         print_str += "{}, ".format(quat_diff_norm[i].item())
    #         print_str += "{}, ".format(align_reward[i].item())
    #         print_str += "{}, ".format(dist_reward_max[i].item())
    #         print_str += "{}, ".format(cubeA_align_square_hole[i].item())
    #         print_str += "{}, ".format(cubeA_on_square_hole[i].item())
    #         print_str += "{}, ".format(gripper_away_from_cubeA[i].item())
    #         print_str += "{}, ".format(stack_reward[i].item())
    #         print_str += "{}, ".format(rewards[i].item())
    #         print(print_str)
    # Compute resets (reset reward?)
    #torch.where(condition, input, other)
    # if condition == True, return input
    # if condition == False, return other
    # if (reached max episode length or stack_reward == True), return empty reset buffer, otherwise, return current reset_buffer)
    # reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
