import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
import torch
"""
USING A CUBBY MADE OF LITTLE FRAMES, WORKS BUT NO ROTATION REWARD. 
"""
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




class CubeCubbyTask(VecTask):

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
            "offset_positive": self.cfg["env"]["offset_positive"]
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
        # self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        # self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        # self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

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
        mug_asset_file = "urdf/ycb/025_mug/025_mug.urdf"
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            mug_asset_file = self.cfg["env"]["asset"].get("assetFileNameMug", mug_asset_file)

            # print("IN THIS IF STATEMENT")
            # print("asset root", asset_root)
            # print("franka_asset_file", franka_asset_file)

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
        
        mug_asset = self.gym.load_asset(self.sim, asset_root, mug_asset_file, asset_options)


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
        # cubeB_opts = gymapi.AssetOptions()
        # cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        # cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        # # Create a floating square (target) slightly larger than the cube
        # square_target_width = 2 * self.cubeB_size  # Twice the width of cubeB
        # square_target_height = 2 * self.cubeB_size  # Twice the height of cubeB
        # square_target_thickness = 0.02  # Slightly thicker for stability, adjust as necessary


        # Create the floating square asset
        # square_target_opts = gymapi.AssetOptions()
        # square_target_opts.fix_base_link = True
        # # square_target_asset = self.gym.create_box(self.sim, square_target_width, square_target_thickness, square_target_height, square_target_opts)
        # # Correct the order of dimensions to ensure the square target lies flat in the X-Y plane
        # square_target_asset = self.gym.create_box(self.sim, square_target_width, square_target_height, square_target_thickness, square_target_opts)

        # # Create larger and flatter wall asset
        wall_opts = gymapi.AssetOptions()
        wall_opts.fix_base_link = True

        wall_width = 1.5  # Increase width of the wall
        wall_thickness = 0.05
        wall_height = 1.0  # Flatter wall
        wall_pos = [0.8, 0.0, 1.0]  # Position the wall across from the table but within arm's reach

        wall_asset = self.gym.create_box(self.sim, wall_width, wall_thickness, wall_height, wall_opts)

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

        # # wall 
        wall_start_pose = gymapi.Transform()
        wall_start_pose.p = gymapi.Vec3(*wall_pos)
        # wall_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        wall_start_pose.r = gymapi.Quat(0.0, 0.0, 0.7071, 0.7071)   

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # cubeB_start_pose = gymapi.Transform()
        # cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        # cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        mug_start_pose = gymapi.Transform()
        mug_start_pose.p = gymapi.Vec3(3.0, 0.0, 0.0)
        mug_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + offset    # 1 for table, table stand, cubeA, cubeB, wall, mug, + 4 boxes for frames
        max_agg_shapes = num_franka_shapes + offset    # 1 for table, table stand, cubeA, cubeB, wall, mug


        # # Define start pose for wall (opposite the table)
        # wall_start_pose = gymapi.Transform()
        # wall_start_pose.p = gymapi.Vec3(0.0, -1.5, 1.0)  # Adjust Y position based on your table position

        self.frankas = []
        self.envs = []


        # Define cubby parameters
        self.frame_bar_length = 0.5  # Length of the bars forming the frame
        self.frame_bar_thickness = .2  # Thickness of the bars forming the frame
        num_frames = 1  # Number of frames to stack
        frame_spacing = 0.3  # Vertical spacing between frames
        table_height = 1
        
        # print("z height for cubby; ", table_height + table_thickness/2  + self.frame_bar_length/2)
        cubby_center = [.3, 0, table_height + table_thickness + self.frame_bar_length/2]  # Cubby position
        cubby_center_tensor = torch.tensor(cubby_center, device=self.device, dtype=torch.long)

        cubby_bottom = torch.tensor([cubby_center[0], cubby_center[1], cubby_center[2] - self.frame_bar_length/2 + self.frame_bar_thickness/2],  device=self.device, dtype=torch.long)
        self.target_cube_center = torch.tensor([cubby_center[0], cubby_center[1], cubby_center[2] - self.frame_bar_length/2 + self.frame_bar_thickness/2 + self.cubeA_size],  device=self.device, dtype=torch.long)
        # print("PRINTING SELF.target_cube_center")
        # print(self.target_cube_center.shape)
        # print(self.target_cube_center)
        # Create environments
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

            # X. Mug
            mug_actor = self.gym.create_actor(env_ptr, mug_asset, mug_start_pose, "mug", i, 0, 0)

            # y.cubby
            cubby_rotation = gymapi.Quat.from_euler_zyx(0, 1.5708, 0)  # 90 degrees around Y-axis
            self.create_cubby(env_ptr, cubby_center, self.frame_bar_length, self.frame_bar_thickness, num_frames, frame_spacing)

            # 2. Table
            # print("table", self.aggregate_mode)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)
            # 3. Wall
            # print("wall", self.aggregate_mode)
            # wall_actor = self.gym.create_actor(env_ptr, wall_asset, wall_start_pose, "wall", i, 5, 0)


            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create cubes
            # 4. Cube A
            # print("cubeA", self.aggregate_mode)
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            
            # 5. Cube B
            # print("cubeB", self.aggregate_mode)
            # self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            # self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            # print("SELF.ENVS", self.envs)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        # self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

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
            "cubby_bottom_handle_{}".format(0): self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubby_bottom_ids[0], "box"),
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
        # self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # 
        self._cubby_bottom_state = self._root_state[:, self._cubby_bottom_ids[0], :]

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            # "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
            "frame_bar_thickness": torch.ones_like(self._eef_state[:, 0]) * self.frame_bar_thickness,
            "frame_bar_length": torch.ones_like(self._eef_state[:, 0]) * self.frame_bar_length
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
        print("Global indices for environment 0:", self._global_indices[0])
        
    def _update_states(self):
        
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],            
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "eef_state": self._eef_state[:, :3],
            # center of a to center of b?
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            # "cubeA_to_cubby": self._cubeA_state[:, :3] - self.target_cube_center,
            "cubby_bottom_pos": self._cubby_bottom_state[:, :3],
            "cubeA_to_cubby": self._cubeA_state[:, :3] - self._cubby_bottom_state[:, :3],
            "cubeA_vel": self._rigid_body_state[:, self._cubeA_id, 7:10],  # Linear velocity (assuming index 7:10 is linear velocity)
            "cubeA_ang_vel": self._rigid_body_state[:, self._cubeA_id, 10:13],  # Angular velocity (assuming index 10:13 is angular velocity)
            # "target_cube_center": self.target_cube_center
        })
        # print("cubeA, rot", self._cubeA_state[:, 3:7])
        # print("_cubby_bottom_state rot: ", self._cubby_bottom_state[:, 3:7])
        #print("STATES")
        #print("_cubeA_state", self._cubeA_state.shape, "state in dict", self._cubeA_state[:, :3])
        #print("_cubeB_state", self._cubeB_state.shape, "state in dict", self._cubeB_state[:, :3])
        # print("cubeA_to_cubeB_pos", self._cubeA_to_cubeB_pos.shape, "state in dict", self._cubeA_to_cubeB_pos[:, :3])
        #print(" END STATES")

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
        # obs include: cubeA_pose (7) + cubeA_to_cubby (3) + eef_pose (7) + q_gripper (2)
        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubby", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        print("FUNC: RESET_IDX()")
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # if not self._i:
        # self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
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
        # print("type env_ids", type(env_ids))
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        # print("multi_env_ids_int32", multi_env_ids_int32)
        # print("env_ids 1", env_ids)

        # print("Shape of _global_indices:", self._global_indices.shape)
        # print("Shape of env_ids:", env_ids.shape)
        # print("Shape of _pos_control:", self._pos_control.shape)
        # print("Shape of _effort_control:", self._effort_control.shape)
        # print("Shape of _root_state:", self._root_state.shape)

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
    
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -3:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        Simple method to reset @cube's position based on fixed initial setup.
        Populates the appropriate self._init_cubeX_state for cube A.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Not used anymore since position is fixed.
        """
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            cube_heights = self.states["cubeA_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A'; got: {cube}")

        # Define the fixed position for cube A, centered on the table
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # Set fixed X and Y values for cube A (centered position)
        sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0)

        # Set the Z value (fixed height)
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights[env_ids] / 2

        # Initialize rotation with no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If you still want to add rotational noise, keep this part
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Set the new initial state for cube A
        this_cube_state_all[env_ids, :] = sampled_cube_state



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
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    frame_bar_length = states["frame_bar_length"]
    frame_bar_thickness = states["frame_bar_thickness"]
    cubeA_size = states["cubeA_size"] #shape = (num_envs)
    cubby_bottom_pos = states["cubby_bottom_pos"] #shape=(num_envs)
    
    # print(f"    cubby_bottom_pos: {cubby_bottom_pos[0].cpu().numpy().tolist()}")

    # print(f"    cubby_bottom_pos: [{cubby_bottom_pos[0, 0].item()}, {cubby_bottom_pos[0, 1].item()}, {cubby_bottom_pos[0, 2].item()}]")

    # print("cubby_bottom_pos", cubby_bottom_pos)
    # print("cubby_bottom_pos.shape", cubby_bottom_pos.shape)

    # print("cubeA_size", cubeA_size)
    # print("cubeA_size.shape", cubeA_size.shape)

    # print("states[cubby_bottom_pos]", states["cubby_bottom_pos"])
    target_height = states["cubby_bottom_pos"][:, 2] + frame_bar_thickness/2 + cubeA_size/2
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
    
    #rotation reward
    # goal_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.5708)  # Rotate 90 degrees about Y-axis


    # how closely aligned cubeA is to cubby (only provided if cubeA is lifted)
    offset = torch.zeros_like(states["cubeA_to_cubby"])
    offset[:, 2] = (cubeA_size + frame_bar_thickness) / 2
    # we used tojust always add offset
    if reward_settings["offset_positive"] == 1:
        d_ab = torch.norm(states["cubeA_to_cubby"] + offset, dim=-1)
    else:
        d_ab = torch.norm(states["cubeA_to_cubby"] - offset, dim=-1)
    align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

    # Dist reward is maximum of dist and align reward
    dist_reward_max = torch.max(dist_reward, align_reward)


    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    cubeA_align_cubby = (torch.norm(states["cubeA_to_cubby"][:, :2], dim=-1) < 0.02)

    # cubeA_on_cubby reward is relative to the table and cubeA_height is relative to table
    cubeA_on_cubby = torch.abs(cubeA_height_from_ground - target_height) < 0.02
    gripper_away_from_cubeA = (d > 0.04)
    # stack reward is a boolean condition that tells us if we have succeeded
    stack_reward = cubeA_align_cubby & cubeA_on_cubby & gripper_away_from_cubeA
    # Compose rewards

    # We either provide the stack reward or the align + dist reward
    # if stack_rewared == True, return r_stack_scale*(1), else return r_dist_scale*dist_reward + r_lift_scale*lift_reward + r_align_scale*align_reward
    rewards = torch.where(
        # if stackrewrd is true
        stack_reward,
        #16
        reward_settings["r_stack_scale"] * stack_reward,
        # otherwise if stack reward is false
        reward_settings["r_dist_scale"] * dist_reward_max + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
            "r_align_scale"] * align_reward,
    )

    # print("stack_reward", stack_reward.shape) #16384 minibatch size

    # Print which conditions are true for which environment
    # does this need to be chagned? or gripper_away_from_cubeA[i].item() TO DO
    # for i in range(stack_reward.shape[0]):
    #     if cubeA_align_cubby[i].item() or cubeA_on_cubby[i].item() or cubeA_lifted[i].item():
    #         print(f"Env {i}:")
    #         print(f"  cubeA_lifted:     {cubeA_lifted[i].item()}")
    #         print(f"  cubeA_align_cubby:    {cubeA_align_cubby[i].item()}")
    #         print(f"  cubeA_on_cubby:   {cubeA_on_cubby[i].item()} cubeA_on_cubby=", 
    #                 f"{cubeA_height_from_ground[i].item()} - {target_height[i].item()} < 0.02")
    #         print(f"  gripper_away_from_cubeA:  {gripper_away_from_cubeA[i].item()}")

    #         # Print cubeA_size as a scalar value
    #         print(f"    cubeA_size: {cubeA_size[i].item()}")
    #         # Print cubeA_pos as a list of coordinates
    #         print(f"    cubeA_pos: [{states['cubeA_pos'][i, 0].item()}, {states['cubeA_pos'][i, 1].item()}, {states['cubeA_pos'][i, 2].item()}]")
    #         # Print eef_lf_pos as a list of coordinates
    #         print(f"    eef_lf_pos: [{states['eef_lf_pos'][i, 0].item()}, {states['eef_lf_pos'][i, 1].item()}, {states['eef_lf_pos'][i, 2].item()}]")
    #         # Print eef_rf_pos as a list of coordinates
    #         print(f"    eef_rf_pos: [{states['eef_rf_pos'][i, 0].item()}, {states['eef_rf_pos'][i, 1].item()}, {states['eef_rf_pos'][i, 2].item()}]")
    #         print(f"     eef_state: [{states['eef_state'][i, 0 ].item()},{states['eef_state'][i, 1 ].item()},{states['eef_state'][i, 2].item()}]")
    #         # Print cubeA_pos_relative as a list of coordinates
    #         print(f"    cubeA_pos_relative: [{states['cubeA_pos_relative'][i, 0].item()}, {states['cubeA_pos_relative'][i, 1].item()}, {states['cubeA_pos_relative'][i, 2].item()}]")
    #         # Print cubby_bottom_pos as a list of coordinates
    #         print(f"    cubby_bottom_pos: [{cubby_bottom_pos[i, 0].item()}, {cubby_bottom_pos[i, 1].item()}, {cubby_bottom_pos[i, 2].item()}]")
    #         # Print target height
    #         print(f"    target_height: {target_height[i].item()}")
    #         # Print frame_bar_length and frame_bar_thickness
    #         print(f"    frame_bar_length: {frame_bar_length[i].item()}")
    #         print(f"    frame_bar_thickness: {frame_bar_thickness[i].item()}")
    #         # Print linear velocity
    #         print(f"    cubeA_linear_velocity: [{states['cubeA_vel'][i, 0].item()}, {states['cubeA_vel'][i, 1].item()}, {states['cubeA_vel'][i, 2].item()}]")

    #         # Print angular velocity
    #         print(f"    cubeA_angular_velocity: [{states['cubeA_ang_vel'][i, 0].item()}, {states['cubeA_ang_vel'][i, 1].item()}, {states['cubeA_ang_vel'][i, 2].item()}]")




    if progress_buf[0].item() == (max_episode_length-1):
        for i in range(1):
            # [progress_buff, ]
            print_str = ""
            print_str += "id: {}, ".format(i)
            print_str += "{}, ".format(progress_buf[i].item())
            print_str += "{}, ".format(dist_reward[i].item())
            print_str += "{}, ".format(cubeA_lifted[i].item())
            print_str += "{}, ".format(align_reward[i].item())
            print_str += "{}, ".format(dist_reward_max[i].item())
            print_str += "{}, ".format(cubeA_align_cubby[i].item())
            print_str += "{}, ".format(cubeA_on_cubby[i].item())
            print_str += "{}, ".format(gripper_away_from_cubeA[i].item())
            print_str += "{}, ".format(stack_reward[i].item())
            print_str += "{}, ".format(rewards[i].item())
            print(print_str)
    # Compute resets (reset reward?)
    #torch.where(condition, input, other)
    # if condition == True, return input
    # if condition == False, return other
    # if (reached max episode length or stack_reward == True), return empty reset buffer, otherwise, return current reset_buffer)
    # reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
