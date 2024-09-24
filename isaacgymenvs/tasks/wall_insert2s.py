import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
import torch
import random


def print_collision_info(gym, sim, actor_handle, env_ptr, actor_name):
    # Retrieve the rigid shape properties for the given actor
    shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
    
    # Iterate through each shape and print the collision group and filter
    for i, shape_prop in enumerate(shape_props):
        print(f"Actor: {actor_name}, Shape Index: {i}")
        # print(f"  Collision Group: {shape_prop._collision_group}")
        print(f"  Filter Mask: {shape_prop.filter}")



offset = 11
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




class WallInsert2S(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        print("IN INIT")
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
            'r_hex_dist_scale': self.cfg["env"]["hex_distRewardScale"],
            'r_hex_lift_scale': self.cfg["env"]["hex_liftRewardScale"],
            'r_hex_align_scale': self.cfg["env"]["hex_alignRewardScale"],
            'r_hex_stack_scale': self.cfg["env"]["hex_stackRewardScale"],

        }

        # self.hex_is_box = self.cfg["env"]["hex_is_box"]
        self.train_objects = self.cfg["env"]['obj_list']
        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        print("self.control_type", self.control_type)
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cube_pos_xyze (7) + cube_to_cubby (3) + eef_pose (7) + q_gripper (2)
        # self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 26
        self.cfg["env"]["numObservations"] = 29 if self.control_type == "osc" else 36

        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        self._init_cube_state = None           # Initial state of obj for the current env
        self._cube_state = None                # Current state of obj for the current env
        self._cube_id = None                   # Actor ID corresponding to obj for a given env
 
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
        self.global_episode_counter = 0  # Initialize step counter

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)


        # print("FINISHED CALL TO SUPER")
        self.episode_success_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Per-episode counter
        self.global_episode_counter = torch.tensor(0,  device=self.device, dtype=torch.float32)

        self.current_object_name = random.sample(self.train_objects, k=1)[0]
        # print("self.current_object_name", self.current_object_name)
        # print("self.current_object_name", self.current_object_name)

        self.cube_task_complete_flag = False
        self.objects_completed = {"objects": []} #torch.tensor([], dtype=torch.float32)  # Or any dtype you need
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



        # Refresh tensors
        self._refresh()

        # Reset all environments
        print("CALLING RESET INDEX")
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        cube_pos_xyz = self.states["cube_pos_xyz"]
        print(f"  cube_pos_xyz:  [{cube_pos_xyz[0, 0].item()}, {cube_pos_xyz[0, 1].item()}, {cube_pos_xyz[0, 2].item()}]")
        hex_pos_xyz = self.states["hex_pos_xyz"]

        print(f"  hex_pos_xyz:  [{hex_pos_xyz[0, 0].item()}, {hex_pos_xyz[0, 1].item()}, {hex_pos_xyz[0, 2].item()}]")


        if self.viewer != None:
            self._set_viewer_params()


    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-2.0, -1.0, 2.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 1.5)

        gymapi.CameraProperties.use_collision_geometry = True
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        intensity = gymapi.Vec3(0.8, 0.8, 0.8)
        ambient = gymapi.Vec3(0.8, 0.8, 0.8)
        self.gym.set_light_parameters(self.sim, 4, intensity, ambient, cam_target)
        
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        # self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(0.8, 0.8, 0.8), gymapi.Vec3(1, 2, 3))

        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        print("IN CREATE ENV")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        wall_asset_files_dict = {"hex_bottom": "urdf/horizontal_wall_shapes/hex_piece.urdf",
                                "square_bottom": "urdf/horizontal_wall_shapes/cube_piece.urdf",
                                "wall_top": "urdf/horizontal_wall_shapes/base_piece.urdf",
                                "wall_bottom": "urdf/horizontal_wall_shapes/base_piece.urdf",
                                "triangle_bottom": "urdf/horizontal_wall_shapes/triangle_piece.urdf"

        }
        shape_asset_files_dict = { "hex_05_15": "urdf/separate_wall_shapes_small/shape_objects/object_hex_05_15.urdf",
                                    "triangle_05_15": "urdf/separate_wall_shapes_small/shape_objects/object_triangle_05_15.urdf",
                                    "cube_05_15": "urdf/separate_wall_shapes_small/shape_objects/object_cube_05_15.urdf",
                                    # "cube_05_15": "urdf/separate_wall_shapes_small/shape_objects/object_hex_05_15.urdf"
                                    # "hex_05_15": "urdf/separate_wall_shapes_small/shape_objects/object_hex_05_15.urdf",
        }
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")        
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
               # wall stuff 
        self.table_height = 1
        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

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
        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

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

        shape_start_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.5708)

        # Create hex object asset
        self.cube_height = 0.05
        self.cube_length = 0.15
        cube_opts = gymapi.AssetOptions()
        # cube_opts.flip_visual_attachments = True       # Ensure correct visual attachment orientation
        cube_opts.fix_base_link = False                # Allow the hexagon to move (not fixed to the base)
        cube_opts.collapse_fixed_joints = False        # Maintain any joints in the asset structure if applicable
        cube_opts.disable_gravity = False              # Enable gravity to make the object fall naturally unless controlled
        cube_opts.thickness = 0.001                    # Specify mesh thickness if needed for accuracy
        # cube_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS   # Enable position control for the gripper or joint actuation
        cube_opts.use_mesh_materials = True            # Use materials from the mesh for accurate physics
        cube_opts.vhacd_enabled = True                 # Enable convex hull decomposition for better collision handling
        # cube_opts.vhacd_params = gymapi.VhacdParams()  # Use VHACD for detailed collision geometry
        # cube_opts.vhacd_params.resolution = 1000000    # Adjust VHACD resolution for accurate collision
        cube_asset = self.gym.load_asset(self.sim, asset_root, shape_asset_files_dict["cube_05_15"], cube_opts)
        # cube_asset = self.gym.create_box(self.sim, *([self.cube_height, self.cube_height, self.cube_length]), cube_opts)
        cube_color = gymapi.Vec3(1.0, 0.0, 0.0)
        cube_start_pose = gymapi.Transform()
        self.cube_start_pose_xyz = [0, -0.15, self.table_height + table_thickness/2 + self.cube_height/2]
        cube_start_pose.p = gymapi.Vec3(*self.cube_start_pose_xyz )
        cube_start_pose.r = shape_start_quat

        # Create hex object asset
        self.hex_length = 0.15
        hex_opts = gymapi.AssetOptions()
        # hex_opts.flip_visual_attachments = True       # Ensure correct visual attachment orientation
        hex_opts.fix_base_link = False                # Allow the hexagon to move (not fixed to the base)
        hex_opts.collapse_fixed_joints = False        # Maintain any joints in the asset structure if applicable
        hex_opts.disable_gravity = False              # Enable gravity to make the object fall naturally unless controlled
        hex_opts.thickness = 0.001                    # Specify mesh thickness if needed for accuracy
        # hex_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS   # Enable position control for the gripper or joint actuation
        hex_opts.use_mesh_materials = True            # Use materials from the mesh for accurate physics
        hex_opts.vhacd_enabled = True                 # Enable convex hull decomposition for better collision handling
        # hex_opts.vhacd_params = gymapi.VhacdParams()  # Use VHACD for detailed collision geometry
        # hex_opts.vhacd_params.resolution = 1000000    # Adjust VHACD resolution for accurate collision
       
        self.hex_height = 0.05
        hex_asset = self.gym.load_asset(self.sim, asset_root, shape_asset_files_dict["hex_05_15"], hex_opts)
        # hex_asset = self.gym.create_box(self.sim, *([self.hex_height, self.hex_height, self.hex_length]), hex_opts)
        hex_color = gymapi.Vec3(0.0, 0.4, 0.1)
        hex_start_pose = gymapi.Transform()
        self.hex_start_pose_xyz = [0.0, 0.0, self.table_height + table_thickness/2 + self.hex_height/2]
        hex_start_pose.p = gymapi.Vec3(*self.hex_start_pose_xyz)
        hex_start_pose.r = shape_start_quat

        # Create triangle object asset
        self.triangle_height = 0.05
        self.triangle_length = 0.15
        triangle_opts = gymapi.AssetOptions()
        triangle_opts.fix_base_link = False                # Allow the hexagon to move (not fixed to the base)
        triangle_opts.collapse_fixed_joints = False        # Maintain any joints in the asset structure if applicable
        triangle_opts.disable_gravity = False  
        triangle_opts.vhacd_enabled = True             # Enable gravity to make the object fall naturally unless controlled
        triangle_opts.thickness = 0.001                    # Specify mesh thickness if needed for accuracy
        # triangle_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS   # Enable position control for the gripper or joint actuation
        triangle_opts.use_mesh_materials = True       
        triangle_asset = self.gym.load_asset(self.sim, asset_root, shape_asset_files_dict["triangle_05_15"], triangle_opts)
        triangle_color = gymapi.Vec3(1, 0.141, 0.922)
        triangle_start_pose = gymapi.Transform()
        self.triangle_start_pose_xyz = [0.0, .2, self.table_height + table_thickness/2 + self.triangle_height/2]
        triangle_start_pose.p = gymapi.Vec3(*self.triangle_start_pose_xyz)
        triangle_start_pose.r = shape_start_quat


        # Create Wall Asset
        # Manually nter some of the wall dimensions
        self._wall_dims_dict = {}
        self._wall_dims_dict["wall_thickness"] = 0.25
        self._wall_dims_dict["wall_height"] = 1.0
        self._wall_dims_dict["wall_width"] = 0.5

        # wall bottom
        self._wall_dims_dict["wall_bottom_z_offset_from_table"] = .1
        self._wall_dims_dict["wall_bottom_z_offset_from_hole"] = 0.1


        # square_bottom
        # center of square piece is right at the surface of the square shelf
        # it is .3 high from the base of the square piece
        # the total height of the square piece is .45
        self._wall_dims_dict["square_bottom_z_offset_from_table"] = .3
        self._wall_dims_dict["square_bottom_y_offset_from_base_center"] = 0.0
        self._wall_dims_dict["square_bottom_z_offset_from_hole"] = 0.0

        # hex_bottom
        # center of hex piece is at the surface of the hex shelf
        # it is .1 aboe the base of the hex piece
        # the total height of the hex piece is .25
        self._wall_dims_dict["hex_bottom_z_offset_from_table"] = .3
        self._wall_dims_dict["hex_bottom_y_offset_from_base_center"] = -.2
        self._wall_dims_dict["hex_bottom_z_offset_from_hole"] = 0.0

        # triangle_bottom
        # center of the triangle piece is at the peak point of the triangle 
        # it is .1 above the base of the triangle piece
        # the total height of the triangle piece is .25
        self._wall_dims_dict["triangle_bottom_z_offset_from_table"] = .3
        self._wall_dims_dict["triangle_bottom_y_offset_from_base_center"] = 0.2
        self._wall_dims_dict["triangle_bottom_z_offset_from_hole"] = 0.0

        # wall_top
        # center of the wall top piece is at the center of the rectangle
        # it is .075 above the base of the wall top piece
        # the total height of the wall top piece is .15
        self._wall_dims_dict["wall_top_z_offset_from_table"] = 0.45+.1
        self._wall_dims_dict["wall_top_z_offset_from_hole"] = -0.1


        self.wall_base_pose = [0.3, 0, self.table_height + table_thickness/2]


        # 2. Custom Wall
        # wall bottom 
                # wall top
        wall_bottom_opts = gymapi.AssetOptions()
        wall_bottom_opts.use_mesh_materials = True  
        wall_bottom_opts.disable_gravity = True
        wall_bottom_opts.fix_base_link = True
        wall_bottom_start_xyz = [self.wall_base_pose[0], self.wall_base_pose[1], self.wall_base_pose[2]+self._wall_dims_dict["wall_bottom_z_offset_from_table"]]
        # wall_bottom_start_xyz = [0,0,0]
        wall_bottom_start_pos = gymapi.Transform()
        wall_bottom_start_pos.p = gymapi.Vec3(*wall_bottom_start_xyz)
        wall_bottom_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
        wall_bottom_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['wall_bottom'], wall_bottom_opts)

        # square_cubby
        square_bottom_wall_opts = gymapi.AssetOptions()
        square_bottom_wall_opts.use_mesh_materials = True  
        square_bottom_wall_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
        square_bottom_wall_opts.override_com = True 
        square_bottom_wall_opts.override_inertia = True 
        square_bottom_wall_opts.disable_gravity = True
        square_bottom_wall_opts.fix_base_link = True
        square_bottom_wall_opts.vhacd_enabled = True 
        square_bottom_wall_opts.vhacd_params = gymapi.VhacdParams() 
        square_bottom_wall_opts.vhacd_params.resolution = 30000000
        square_bottom_wall_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        square_bottom_start_xyz = [self.wall_base_pose[0], self.wall_base_pose[1]+self._wall_dims_dict["square_bottom_y_offset_from_base_center"], self.wall_base_pose[2]+self._wall_dims_dict["square_bottom_z_offset_from_table"]]
        square_bottom_start_pos = gymapi.Transform()
        square_bottom_start_pos.p = gymapi.Vec3(*square_bottom_start_xyz)
        square_bottom_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
        square_bottom_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['square_bottom'], square_bottom_wall_opts)


        # hex bottom
        hex_bottom_wall_opts = gymapi.AssetOptions()
        hex_bottom_wall_opts.use_mesh_materials = True  
        hex_bottom_wall_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
        hex_bottom_wall_opts.override_com = True 
        hex_bottom_wall_opts.override_inertia = True 
        hex_bottom_wall_opts.disable_gravity = True
        hex_bottom_wall_opts.fix_base_link = True
        hex_bottom_wall_opts.vhacd_enabled = True 
        hex_bottom_wall_opts.vhacd_params = gymapi.VhacdParams() 
        hex_bottom_wall_opts.vhacd_params.resolution = 30000000
        hex_bottom_wall_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        hex_bottom_start_xyz = [self.wall_base_pose[0], self.wall_base_pose[1]+self._wall_dims_dict["hex_bottom_y_offset_from_base_center"], self.wall_base_pose[2]+self._wall_dims_dict["hex_bottom_z_offset_from_table"]]
        # hex_bottom_start_xyz = [0,0,0]

        hex_bottom_start_pos = gymapi.Transform()
        hex_bottom_start_pos.p = gymapi.Vec3(*hex_bottom_start_xyz)
        hex_bottom_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
        hex_bottom_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['hex_bottom'], hex_bottom_wall_opts)

        # triangle bottom
        triangle_bottom_wall_opts = gymapi.AssetOptions()
        triangle_bottom_wall_opts.use_mesh_materials = True  
        triangle_bottom_wall_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
        triangle_bottom_wall_opts.override_com = True 
        triangle_bottom_wall_opts.override_inertia = True 
        triangle_bottom_wall_opts.disable_gravity = True
        triangle_bottom_wall_opts.fix_base_link = True
        triangle_bottom_wall_opts.vhacd_enabled = True 
        triangle_bottom_wall_opts.vhacd_params = gymapi.VhacdParams() 
        triangle_bottom_wall_opts.vhacd_params.resolution = 30000000
        triangle_bottom_wall_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        triangle_bottom_start_xyz = [self.wall_base_pose[0], self.wall_base_pose[1]+self._wall_dims_dict["triangle_bottom_y_offset_from_base_center"], self.wall_base_pose[2]+self._wall_dims_dict["triangle_bottom_z_offset_from_table"]]
        # triangle_bottom_start_xyz = [0,0,0]

        triangle_bottom_start_pos = gymapi.Transform()
        triangle_bottom_start_pos.p = gymapi.Vec3(*triangle_bottom_start_xyz)
        triangle_bottom_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
        triangle_bottom_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['triangle_bottom'], triangle_bottom_wall_opts)


        # wall top
        wall_top_opts = gymapi.AssetOptions()
        wall_top_opts.use_mesh_materials = True  
        wall_top_opts.disable_gravity = True
        wall_top_opts.fix_base_link = True
        wall_top_start_xyz = [self.wall_base_pose[0], self.wall_base_pose[1], self.wall_base_pose[2]+self._wall_dims_dict["wall_top_z_offset_from_table"]]
        # wall_top_start_xyz = [0,0,0]
        wall_top_start_pos = gymapi.Transform()
        wall_top_start_pos.p = gymapi.Vec3(*wall_top_start_xyz)
        wall_top_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
        wall_top_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['wall_top'], wall_top_opts)

        # count franka asset nums
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        
        wall_bottom_bodies = self.gym.get_asset_rigid_body_count(wall_bottom_asset)
        wall_bottom_shapes = self.gym.get_asset_rigid_shape_count(wall_bottom_asset)
        # count wall asset nums
        square_bottom_bodies = self.gym.get_asset_rigid_body_count(square_bottom_asset)
        square_bottom_shapes = self.gym.get_asset_rigid_shape_count(square_bottom_asset)

        hex_bottom_bodies = self.gym.get_asset_rigid_body_count(hex_bottom_asset)
        hex_bottom_shapes = self.gym.get_asset_rigid_shape_count(hex_bottom_asset)

        triangle_bottom_bodies = self.gym.get_asset_rigid_body_count(triangle_bottom_asset)
        triangle_bottom_shapes = self.gym.get_asset_rigid_shape_count(triangle_bottom_asset)

        wall_top_bodies = self.gym.get_asset_rigid_body_count(wall_top_asset)
        wall_top_shapes = self.gym.get_asset_rigid_shape_count(wall_top_asset)

        # count shape object asset nums
        cube_bodies = self.gym.get_asset_rigid_body_count(cube_asset)
        cube_shapes = self.gym.get_asset_rigid_shape_count(cube_asset)

        hex_bodies = self.gym.get_asset_rigid_body_count(hex_asset)
        hex_shapes = self.gym.get_asset_rigid_shape_count(hex_asset)
        
        triangle_bodies = self.gym.get_asset_rigid_body_count(triangle_asset)
        triangle_shapes = self.gym.get_asset_rigid_shape_count(triangle_asset)

        # count table asset nums
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        
        num_table_stand_bodies = self.gym.get_asset_rigid_body_count(table_stand_asset)
        num_table_stand_shapes = self.gym.get_asset_rigid_shape_count(table_stand_asset)


        max_agg_bodies = num_franka_bodies + square_bottom_bodies + hex_bottom_bodies \
                        + triangle_bottom_bodies + wall_top_bodies + num_table_bodies \
                        +  num_table_stand_bodies + cube_bodies  + hex_bodies \
                        + triangle_bodies + wall_bottom_bodies
        max_agg_shapes = num_franka_shapes + square_bottom_shapes + hex_bottom_shapes \
                        + triangle_bottom_shapes + wall_top_shapes + num_table_shapes \
                        + num_table_stand_shapes + cube_shapes + hex_shapes \
                        + triangle_shapes + wall_bottom_shapes

        # Print the number of rigid bodies and shapes for each asset
        print(f"Franka: Bodies: {num_franka_bodies}, Shapes: {num_franka_shapes}")
        print(f"Square Bottom: Bodies: {square_bottom_bodies}, Shapes: {square_bottom_shapes}")
        print(f"Hex Bottom: Bodies: {hex_bottom_bodies}, Shapes: {hex_bottom_shapes}")
        print(f"Wall Top: Bodies: {wall_top_bodies}, Shapes: {wall_top_shapes}")

        print(f"Table: Bodies: {num_table_bodies}, Shapes: {num_table_shapes}")
        print(f"Table Stand: Bodies: {num_table_stand_bodies}, Shapes: {num_table_stand_shapes}")
        print(f"obj A: Bodies: {cube_bodies}, Shapes: {cube_shapes}")
        print(f"Hexagon: Bodies: {hex_bodies}, Shapes: {hex_shapes}")
        print(f"Triangle: Bodies: {triangle_bodies}, Shapes: {triangle_shapes}")


        self.frankas = []
        self.envs = []
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
            self._wall_bottom_id = self.gym.create_actor(env_ptr, wall_bottom_asset, wall_bottom_start_pos, "wall_bottom", i, 1, 0)

            self._square_bottom_id = self.gym.create_actor(env_ptr, square_bottom_asset, square_bottom_start_pos, "square_bottom", i, 1, 0)

            # hex bottom
            self._hex_bottom_id = self.gym.create_actor(env_ptr, hex_bottom_asset, hex_bottom_start_pos, "hex_bottom", i, 1, 0)

            # triangle bottom
            self._triangle_bottom_id = self.gym.create_actor(env_ptr, triangle_bottom_asset, triangle_bottom_start_pos, "triangle_bottom", i, 1, 0)

            # wall top
            self._wall_top_id = self.gym.create_actor(env_ptr, wall_top_asset, wall_top_start_pos, "wall_top", i, 1, 0)

            # 3. Table
            self._table_id = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            self._table_stand_id = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 4. Create Objects
            # make sure you create obj last because something is hardcoded in a reset
            self._triangle_id = self.gym.create_actor(env_ptr, triangle_asset, triangle_start_pose, "triangle", i, 4, 0)
            self.gym.set_rigid_body_color(env_ptr, self._triangle_id, 0, gymapi.MESH_VISUAL, triangle_color)
            triangle_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._triangle_id)[0]
            
            self._hex_id = self.gym.create_actor(env_ptr, hex_asset, hex_start_pose, "hex", i, 4, 0)
            self.gym.set_rigid_body_color(env_ptr, self._hex_id, 0, gymapi.MESH_VISUAL, hex_color)
            hex_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._hex_id)[0]
            # print("hex mass", hex_props.mass)

            self._cube_id = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 4, 0)
            self.gym.set_rigid_body_color(env_ptr, self._cube_id, 0, gymapi.MESH_VISUAL, cube_color)
            cube_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._cube_id)[0]
            # print("obj mass", props.mass)



            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            
            # Store the created env pointers
            self.envs.append(env_ptr)
            # print("SELF.ENVS", self.envs)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_cube_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_hex_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_triangle_state = torch.zeros(self.num_envs, 13, device=self.device)

        obj_goal_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.5708)
        self._desired_obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_obj_quat[:, 0] = obj_goal_quat.x
        self._desired_obj_quat[:, 1] = obj_goal_quat.y
        self._desired_obj_quat[:, 2] = obj_goal_quat.z
        self._desired_obj_quat[:, 3] = obj_goal_quat.w
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
        # object states
        self._cube_state = self._root_state[:, self._cube_id, :]
        self._hex_state = self._root_state[:, self._hex_id, :]
        self._triangle_state = self._root_state[:, self._triangle_id, :]

        # wall piece states
        self._square_bottom_state = self._root_state[:, self._square_bottom_id, :]
        self._hex_bottom_state = self._root_state[:, self._hex_bottom_id, :]
        # self._hex_z_offset = np.zeros_like(self._hex_bottom_state)
        # self._hex_z_offset[:, 2] = np.zeros_like(self._hex_bottom_state)
        self._hex_bottom_state_center = self._hex_bottom_state
        self._triangle_bottom_state = self._root_state[:, self._triangle_bottom_id, :]
        self._wall_top_state = self._root_state[:, self._wall_top_id, :]

        self._table_state = self._root_state[:, self._table_id, :]
        self._table_stand_state = self._root_state[:, self._table_stand_id, :]
        
        self._cube_length_torch = self.cube_length*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1) 
        self._cube_height_torch = self.cube_height*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1) 
        self._cube_start_y_torch = self.cube_start_pose_xyz[1]*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1) 

        self._triangle_length_torch = self.triangle_length*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1) 
        self._triangle_height_torch = self.triangle_height*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1)  
        self._triangle_start_y_torch = self.triangle_start_pose_xyz[1]*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1) 

        self._hex_length_torch = self.hex_length*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1) 
        self._hex_height_torch = self.hex_height*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1)  
        self._hex_start_y_torch = self.hex_start_pose_xyz[1]*torch.ones_like(self._eef_state[:, 0])#.reshape(-1, 1) 

        self._cube_goal_grip_pos_x = torch.ones_like(self._eef_state[:, :3])
        self._cube_goal_grip_pos_x[:, 0] = -self._cube_length_torch/2
        
        self.states.update({
            "cube_length": self._cube_length_torch,
            "cube_height": self._cube_height_torch,
            "hex_length": self._hex_length_torch,
            "hex_height": self._hex_height_torch,
            "triangle_length": self._triangle_length_torch,
            "triangle_height": self._triangle_height_torch,
            "cube_start_y": self._cube_start_y_torch,
            "hex_start_y": self._hex_start_y_torch,
            "triangle_start_y": self._triangle_start_y_torch,
           
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
            "eef_lf_pos_xyz": self._eef_lf_state[:, :3],
            "eef_rf_pos_xyz": self._eef_rf_state[:, :3],
            "eef_state": self._eef_state[:, :3],

            # Cube 
            "cube_length": torch.ones_like(self._eef_state[:, 0]) * self.cube_length,
            "cube_quat": self._cube_state[:, 3:7],
            "cube_pos_xyz": self._cube_state[:, :3],
            # "length": self._cube_state[:, :3] + self._cube_goal_grip_pos_x[:, :3],
            "cube_goal_grip_pos_to_gripper_xyz": self._cube_state[:, :3] + self._cube_goal_grip_pos_x - self._eef_state[:, :3],
            "cube_to_gripper_xyz": self._cube_state[:, :3] - self._eef_state[:, :3], # obj position relative to hand. How far is obj A from hand.
            "cube_vel": self._rigid_body_state[:, self._cube_id, 7:10],  # Linear velocity (assuming index 7:10 is linear velocity)
            "cube_ang_vel": self._rigid_body_state[:, self._cube_id, 10:13],  # Angular velocity (assuming index 10:13 is angular velocity),
            "cube_to_hole_bottom_xyz": self._cube_state[:, :3] - self._square_bottom_state[:, :3],
        
            # Hex 
            "hex_length": torch.ones_like(self._eef_state[:, 0]) * self.hex_length,
            "hex_quat": self._hex_state[:, 3:7],
            "hex_pos_xyz": self._hex_state[:, :3],
            # "cube_goal_grip_pos_to_gripper_xyz": self._cube_state[:, :3] + self._cube_goal_grip_pos_x - self._eef_state[:, :3],
            "hex_to_gripper_xyz": self._hex_state[:, :3] - self._eef_state[:, :3], # obj position relative to hand. How far is obj A from hand.
            "hex_vel": self._rigid_body_state[:, self._hex_id, 7:10],  # Linear velocity (assuming index 7:10 is linear velocity)
            "hex_ang_vel": self._rigid_body_state[:, self._hex_id, 10:13],  # Angular velocity (assuming index 10:13 is angular velocity),
            "hex_to_hole_bottom_xyz": self._hex_state[:, :3] - self._hex_bottom_state[:, :3],
        
            # obj A 
            "triangle_length": torch.ones_like(self._eef_state[:, 0]) * self.triangle_length,
            "triangle_quat": self._triangle_state[:, 3:7],
            "triangle_pos_xyz": self._triangle_state[:, :3],
            "triangle_to_gripper_xyz": self._triangle_state[:, :3] - self._eef_state[:, :3], # triangle position relative to hand. How far is obj A from hand.
            "triangle_vel": self._rigid_body_state[:, self._triangle_id, 7:10],  # Linear velocity (assuming index 7:10 is linear velocity)
            "triangle_ang_vel": self._rigid_body_state[:, self._triangle_id, 10:13],  # Angular velocity (assuming index 10:13 is angular velocity),
            "triangle_to_hole_bottom_xyz": self._triangle_state[:, :3] - self._triangle_bottom_state[:, :3],
        
            # table
            "table_state": self._table_state[:, :3],
            "table_stand_state": self._table_stand_state[:, :3],

            # Wall
            # xyz positions of the base of each cubby
            "cube_hole_bottom_xyz": self._square_bottom_state[:, :3], # wall center
            "hex_hole_bottom_xyz": self._hex_bottom_state[:, :3],
            "triangle_hole_bottom_xyz": self._triangle_bottom_state[:, :3],

            # Evaluation
            "episode_success_counter": self.episode_success_counter,
            "global_episode_counter": self.global_episode_counter,

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
        # print(self.objects_completed)
        self.rew_buf[:], self.reset_buf[:], self.objects_completed = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length, self.current_object_name, self.objects_completed, self.cube_task_complete_flag
        )
        # print("self.objects_completed", self.objects_completed["objects"])


    # def compute_observations(self):
    #     self._refresh()
    #     obs = ["cube_quat", "cube_pos_xyz", "cube_to_hole_bottom_xyz","hex_quat", "hex_pos_xyz", "hex_to_hole_bottom_xyz",  "eef_pos", "eef_quat"]
        
    #     #"triangle_quat",
        
    #     # "triangle_pos_xyz", "triangle_to_hole_bottom_xyz", "hex_quat", "hex_pos_xyz", "hex_to_hole_bottom_xyz"]

    #     obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
    #     self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)


    def compute_observations(self):
        self._refresh()
        cube_obs = torch.cat([self.states['cube_quat'], self.states['cube_pos_xyz'], self.states['cube_to_hole_bottom_xyz']], dim=-1)
        hex_obs = torch.cat([self.states['hex_quat'], self.states['hex_pos_xyz'], self.states['hex_to_hole_bottom_xyz']], dim=-1)
        if self.control_type == "osc":
            eefs_obs = torch.cat([self.states["eef_pos"], self.states["eef_quat"], self.states["q_gripper"]], dim=-1)
        else:
            eefs_obs = torch.cat([self.states["eef_pos"], self.states["eef_quat"], self.states["q"]], dim=-1)

        # Mask out hex observations if cube task is not complete
        if not self.cube_task_complete_flag:
            hex_obs = torch.zeros_like(hex_obs)
        
        # Concatenate cube and hex observations, making sure the observation size remains fixed
        self.obs_buf = torch.cat([cube_obs, hex_obs, eefs_obs], dim=-1)
        # self.obs_buf = torch.cat([cube_obs, eefs_obs], dim=-1)
    def reset_idx(self, env_ids):
        # print("FUNC: RESET_IDX()")
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        # self.current_object_name = random.sample(["cube", "triangle", "hex"], k=1)
        # choose separate object for each environment
        self.current_object_name = random.sample(self.train_objects, k=1)[0]

        # print("object is: {}".format(self.current_object_name))
        self._reset_init_object_state(obj='cube', env_ids=env_ids, check_valid=True)
        self._reset_init_object_state(obj='hex', env_ids=env_ids, check_valid=True)
        self._reset_init_object_state(obj='triangle', env_ids=env_ids, check_valid=True)

        # Write these new init states to the sim states
        self._cube_state[env_ids] = self._init_cube_state[env_ids]
        self._hex_state[env_ids] = self._init_hex_state[env_ids]
        self._triangle_state[env_ids] = self._init_triangle_state[env_ids]

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
    
        multi_env_ids_objs_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_objs_int32), len(multi_env_ids_objs_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    

    def _reset_init_object_state(self, obj, env_ids, check_valid=True):
        """
        Simple method to reset @obj's position based on fixed initial setup.
        Populates the appropriate self._init_objX_state for obj A.

        Args:
            obj(str): Which obj to sample location for. Either 'obj' or 'cylinder'
            env_ids (tensor or None): Specific environments to reset obj for
            check_valid (bool): Not used anymore since position is fixed.
        """
        # If env_ids is None, we reset all the envs
        # print("IN RESET STATE :{}".format(obj))
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_state = torch.zeros(num_resets, 13, device=self.device)
        y_offsets = torch.zeros(num_resets, device=self.device)

        # Get correct references depending on which one was selected
        init_state_dict = {'cube': self._init_cube_state,
                            'hex': self._init_hex_state,
                            'triangle': self._init_triangle_state}
        start_y_dict = {"cube": self.states["cube_start_y"],
                       "hex": self.states["hex_start_y"],
                       "triangle": self.states["triangle_start_y"]}
        height_dict = {"cube": self.states["cube_height"],
                       "hex": self.states["hex_height"],
                       "triangle": self.states["triangle_height"]}

        obj_name = obj.lower()
        this_obj_state_all = init_state_dict[obj_name]
        obj_height = height_dict[obj_name]
        obj_y_offset = start_y_dict[obj_name]
        
        # Define the fixed position for obj A, centered on the table
        sampled_obj_state = torch.zeros(num_resets, 13, device=self.device)
            # Set the Z value (fixed height)
            # print("obj_h.squeeze(-1)", obj_height.squeeze(-1))
            # print("obj_height.squeeze(-1)[env_ids]", obj_height.squeeze(-1)[env_ids])
        sampled_obj_state[:, 1] = self._table_surface_pos[1] + obj_y_offset[env_ids]

        # Set the Z value (fixed height)
        # print("obj_h.squeeze(-1)", obj_height.squeeze(-1))
        # print("obj_height.squeeze(-1)[env_ids]", obj_height.squeeze(-1)[env_ids])
        sampled_obj_state[:, 2] =  self._table_surface_pos[2] + obj_height[env_ids]/2
        # shape_start_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.5708)
        shape_start_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.5708) # rotate 90 deg around y
        # Initialize rotation with no rotation (quat w = 1)
        # sampled_obj_state[:, 6] = 1.0
        sampled_obj_state[:, 3] = shape_start_quat.x
        sampled_obj_state[:, 4] = shape_start_quat.y
        sampled_obj_state[:, 5] = shape_start_quat.z
        sampled_obj_state[:, 6] = shape_start_quat.w

        
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
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


        # Every 100 episodes, print success statistics

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            obj_pos_xyz = self.states["obj_pos_xyz"]
            obj_rot = self.states["cube_quat"]
            
            # objB_pos = self.states["objB_pos"]
            # objB_rot = self.states["objB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cube_pos_xyz, objB_pos), (eef_rot, cube_rot, objB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])



@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length, object_name, objects_completed, cube_task_complete_flag
):  

    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float, str, Dict[str, List[str]], bool) -> Tuple[Tensor, Tensor, Dict[str, List[str]]]
    
    eef_lf_pos_xyz = states["eef_lf_pos_xyz"]
    eef_rf_pos_xyz = states["eef_rf_pos_xyz"]
    eef_state = states["eef_state"]
    

    ######## CUBE ##################################################################
    cube_length = states["cube_length"]    
    cube_height = states["cube_height"]    
    cube_hole_bottom_xyz = states["cube_hole_bottom_xyz"]
    cube_to_hole_bottom_xyz = states["cube_to_hole_bottom_xyz"]

    target_cube_height = cube_hole_bottom_xyz[:, 2] + cube_height/2
    cube_goal_grip_pos_to_gripper_xyz = states["cube_goal_grip_pos_to_gripper_xyz"]
    cube_goal_grip_pos_xyz = states["cube_goal_grip_pos_xyz"]
    cube_pos_xyz = states["cube_pos_xyz"]

    cube_vel = states[f'{object_name}_vel']
    cube_ang_vel = states[f'{object_name}_ang_vel']

    cube_to_gripper_xyz = states["cube_to_gripper_xyz"]
    # Reward for gripping the obj
    dist_cube_to_gripper = torch.norm(states["cube_to_gripper_xyz"], dim=-1)
    dist_cube_to_left_finger = torch.norm(cube_pos_xyz - eef_lf_pos_xyz, dim=-1)
    dist_cube_to_right_finger = torch.norm(cube_pos_xyz - eef_rf_pos_xyz, dim=-1)
    cube_gripping_reward = 1 - torch.tanh(10.0 * (dist_cube_to_gripper+ dist_cube_to_left_finger + dist_cube_to_right_finger) / 3)

    # Reward for lifting obj
    cube_lifting_moe = 0.04
    cube_height_from_table = cube_pos_xyz[:, 2] - reward_settings["table_height"]
    cube_height_from_ground = cube_pos_xyz[:, 2] 
    cube_lifted = (cube_height_from_table - cube_height) > cube_lifting_moe
    is_cube_lifted = cube_lifted

    # How closely aligned obj is to cubby (only provided if obj is lifted) num_envs X 3 
    
    cube_goal_position_z = torch.zeros_like(cube_hole_bottom_xyz)
    cube_goal_position_z[:, 2] = cube_height/2
    cube_goal_position_xyz = cube_hole_bottom_xyz + cube_goal_position_z

    diff_cube_to_goal = cube_pos_xyz - cube_goal_position_xyz
    # then align to the final z position
    dist_cube_to_goal_z = torch.abs(diff_cube_to_goal[:, 2])
    cube_z_align_reward = (1 - torch.tanh(10.0 * dist_cube_to_goal_z)) * is_cube_lifted 
    cube_z_align_val_flag = diff_cube_to_goal[:, 2] < .05
    cube_z_align_positive_flag = 0 < diff_cube_to_goal[:, 2]
    cube_z_align_flag = cube_z_align_val_flag * cube_z_align_positive_flag

    # then align along the y axis

    # start by getting to y halfway point
    dist_cube_to_goal_y_halfway = torch.abs(cube_pos_xyz[:, 1]-.5*cube_goal_position_xyz[:, 1])
    cube_y_align_halfway_reward = (1 - torch.tanh(10.0 * dist_cube_to_goal_y_halfway)) * is_cube_lifted * cube_z_align_flag
    cube_y_align_halfway_flag = dist_cube_to_goal_y_halfway < .01

    dist_cube_to_goal_y = torch.abs(diff_cube_to_goal[:, 1])
    cube_y_align_reward = (1 - torch.tanh(10.0 * dist_cube_to_goal_y)) * is_cube_lifted  * cube_z_align_flag * cube_y_align_halfway_flag
    cube_y_align_flag = dist_cube_to_goal_y < .05

    # then finally align along the x axis (insertion axis)
    dist_cube_to_goal_x = torch.abs(diff_cube_to_goal[:, 0])
    cube_x_align_reward = (1 - torch.tanh(10.0 * dist_cube_to_goal_x)) * is_cube_lifted * cube_z_align_flag * cube_y_align_flag
    cube_x_align_flag = dist_cube_to_goal_x < .075

    cube_align_reward = cube_z_align_reward + cube_y_align_reward + cube_y_align_halfway_reward + cube_x_align_reward
    cube_aligned_flag = cube_z_align_flag * cube_y_align_flag * cube_x_align_flag
    # dist_cube_to_goal = torch.norm(cube_pos_xyz - cube_goal_position_xyz, dim=-1)

    
    # cube_align_reward = (1 - torch.tanh(10.0 * dist_cube_to_goal)) * is_cube_lifted
    
    # Dist reward is maximum of dist and align reward
    max_reward_bw_gripper_align = torch.max(cube_gripping_reward, cube_align_reward)

    # # obj XY alignment 
    # cube_y_alignment  = torch.abs(cube_to_hole_bottom_xyz[:, 1]) < .05
    # cube_x_alignment = torch.abs(cube_to_hole_bottom_xyz[:, 0]) < .075

    # xy_cube_to_square_hole = cube_to_hole_bottom_xyz[:, :2]
    # dist_cube_to_cube_hole_xy = (torch.norm(xy_cube_to_square_hole, dim=-1) < 0.02)

    # obj Z alignment 
    # dist_cube_to_cube_hole_z = torch.abs(cube_height_from_ground - target_cube_height) < 0.02
    
    # gripper releases obj 
    gripper_released_cube = (dist_cube_to_gripper > 0.08)

    # stack reward is a boolean condition that tells us if we have succeeded
    # cube_placement_complete = cube_x_alignment & cube_y_alignment & dist_cube_to_cube_hole_z & gripper_released_cube
    cube_placement_complete = cube_aligned_flag * gripper_released_cube
    # Update the cube task completion flag if cube is placed correctly
    if cube_placement_complete.any():
        cube_task_complete_flag = True

    # Compose rewards
    scaled_cube_placement_complete = reward_settings["r_stack_scale"] * cube_placement_complete
    scaled_dist_reward_cube = reward_settings["r_dist_scale"] * max_reward_bw_gripper_align
    scaled_lifted_reward_cube = reward_settings["r_lift_scale"] * is_cube_lifted
    scaled_alignment_reward_cube = reward_settings["r_align_scale"] * cube_align_reward
    scaled_mid_reward_cube = scaled_dist_reward_cube + scaled_lifted_reward_cube + scaled_alignment_reward_cube

    # if arg0 is true, arg1, else arg2
    cube_rewards = torch.where(
        cube_placement_complete, 
        scaled_cube_placement_complete,
        scaled_mid_reward_cube
    )


    # ######## HEX ##################################################################
    # Compute hex-related rewards only if the cube task is completed
    if cube_task_complete_flag:
        hex_length = states["hex_length"]    
        hex_height = states["hex_height"]    
        hex_hole_bottom_xyz = states["hex_hole_bottom_xyz"]
        hex_to_hole_bottom_xyz = states["hex_to_hole_bottom_xyz"]

        target_hex_height = hex_hole_bottom_xyz[:, 2] + hex_height/2
        hex_goal_grip_pos_to_gripper_xyz = states["hex_goal_grip_pos_to_gripper_xyz"]
        hex_goal_grip_pos_xyz = states["hex_goal_grip_pos_xyz"]
        hex_pos_xyz = states["hex_pos_xyz"]

        hex_vel = states['hex_vel']
        hex_ang_vel = states['hex_ang_vel']

        hex_to_gripper_xyz = states["hex_to_gripper_xyz"]
        # Reward for gripping the obj
        dist_hex_to_gripper = torch.norm(states["hex_to_gripper_xyz"], dim=-1)
        dist_hex_to_left_finger = torch.norm(hex_pos_xyz - eef_lf_pos_xyz, dim=-1)
        dist_hex_to_right_finger = torch.norm(hex_pos_xyz - eef_rf_pos_xyz, dim=-1)
        hex_gripping_reward = (1 - torch.tanh(10.0 * (dist_hex_to_gripper+ dist_hex_to_left_finger + dist_hex_to_right_finger) / 3)) * cube_placement_complete

        # Reward for lifting obj
        hex_lifting_moe = 0.04
        hex_height_from_table = hex_pos_xyz[:, 2] - reward_settings["table_height"]
        hex_height_from_ground = hex_pos_xyz[:, 2]
        hex_lifted = (hex_height_from_table - hex_height) > hex_lifting_moe
        is_hex_lifted = hex_lifted * cube_placement_complete

        # How closely aligned obj is to cubby (only provided if obj is lifted) num_envs X 3 
    
        hex_goal_position_z = torch.zeros_like(hex_hole_bottom_xyz)
        hex_goal_position_z[:, 2] = hex_height/2
        hex_goal_position_xyz = hex_hole_bottom_xyz + hex_goal_position_z

        diff_hex_to_goal = hex_pos_xyz - hex_goal_position_xyz
        # then align to the final z position
        dist_hex_to_goal_z = torch.abs(diff_hex_to_goal[:, 2])
        hex_z_align_reward = (1 - torch.tanh(10.0 * dist_hex_to_goal_z)) * is_hex_lifted
        hex_z_align_val_flag = diff_hex_to_goal[:, 2] < .05
        hex_z_align_positive_flag = 0 < diff_hex_to_goal[:, 2]
        hex_z_align_flag = hex_z_align_val_flag * hex_z_align_positive_flag

        # then align along the y axis
        # start by getting to y halfway point
        dist_hex_to_goal_y_halfway = torch.abs(hex_pos_xyz[:, 1]-.5*hex_goal_position_xyz[:, 1])
        hex_y_align_halfway_reward = (1 - torch.tanh(10.0 * dist_hex_to_goal_y_halfway)) * is_hex_lifted * hex_z_align_flag
        hex_y_align_halfway_flag = dist_hex_to_goal_y_halfway < .01

        dist_hex_to_goal_y = torch.abs(diff_hex_to_goal[:, 1])
        hex_y_align_reward = (1 - torch.tanh(10.0 * dist_hex_to_goal_y)) * is_hex_lifted  * hex_z_align_flag
        hex_y_align_flag = dist_hex_to_goal_y < .05

        # then finally align along the x axis (insertion axis)
        dist_hex_to_goal_x = torch.abs(diff_hex_to_goal[:, 0])
        hex_x_align_reward = (1 - torch.tanh(10.0 * dist_hex_to_goal_x)) * is_hex_lifted * hex_z_align_flag * hex_y_align_flag
        hex_x_align_flag = dist_hex_to_goal_x < .075

        hex_align_reward = hex_y_align_halfway_reward + hex_z_align_reward + hex_y_align_reward + hex_x_align_reward
        hex_aligned_flag = hex_z_align_flag * hex_y_align_flag * hex_x_align_flag

        # # How closely aligned obj is to cubby (only provided if obj is lifted) num_envs X 3 
        # hex_goal_position_z = torch.zeros_like(hex_hole_bottom_xyz)
        # hex_goal_position_z[:, 2] = hex_height/2
        # hex_goal_position_xyz = hex_hole_bottom_xyz + hex_goal_position_z
        # dist_hex_to_goal = torch.norm(hex_pos_xyz - hex_goal_position_xyz, dim=-1)
        # hex_align_reward = (1 - torch.tanh(10.0 * dist_hex_to_goal)) * is_hex_lifted * hex_placement_complete
        
        # Dist reward is maximum of dist and align reward
        max_reward_bw_gripper_align = torch.max(hex_gripping_reward, hex_align_reward)

        # obj XY alignment 
        # hex_y_alignment  = torch.abs(hex_to_hole_bottom_xyz[:, 1]) < .05
        # hex_x_alignment = torch.abs(hex_to_hole_bottom_xyz[:, 0]) < .075

        # xy_hex_to_square_hole = hex_to_hole_bottom_xyz[:, :2]
        # dist_hex_to_hex_hole_xy = (torch.norm(xy_hex_to_square_hole, dim=-1) < 0.02)

        # obj Z alignment 
        # dist_hex_to_hex_hole_z = torch.abs(hex_height_from_ground - target_hex_height) < 0.02
        
        # gripper releases obj 
        gripper_released_hex = (dist_hex_to_gripper > 0.08)

        # stack reward is a boolean condition that tells us if we have succeeded
        # Hex placement is complete if it's correctly aligned and released, but only after the cube is placed
        # hex_placement_complete = hex_x_alignment & hex_y_alignment & dist_hex_to_hex_hole_z & gripper_released_hex & cube_placement_complete
        hex_placement_complete = hex_aligned_flag * gripper_released_hex

        # Compose rewards
        scaled_hex_placement_complete = reward_settings["r_hex_stack_scale"] * hex_placement_complete
        scaled_dist_reward_hex = reward_settings["r_hex_dist_scale"] * max_reward_bw_gripper_align
        scaled_lifted_reward_hex = reward_settings["r_hex_lift_scale"] * is_hex_lifted
        scaled_alignment_reward_hex = reward_settings["r_hex_align_scale"] * hex_align_reward
        scaled_mid_reward_hex = scaled_dist_reward_hex + scaled_lifted_reward_hex + scaled_alignment_reward_hex
        # if arg0 is true, arg1, else arg2
        hex_rewards = torch.where(
            hex_placement_complete, 
            scaled_hex_placement_complete,
            scaled_mid_reward_hex
        )
        # for i in range(cube_placement_complete.shape[0]):
        #     cube_placement_progress = cube_y_align_flag[i].item() or (dist_cube_to_goal_z[i].item() and cube_lifted[i].item())
        #     # hex_placement_progress = dist_hex_to_hex_hole_xy[i].item() or (dist_hex_to_hex_hole_z[i].item() and hex_lifted[i].item())
        #     if cube_placement_progress: # or hex_placement_progress:
        #         print("HEX INFO")
        #         print(f"  hex_lifted:         {hex_lifted[i].item()}")
        #         print(f"  dist_hex_to_goal_x: {dist_hex_to_goal_x[i].item()}")
        #         print(f"  dist_hex_to_goal_y: {dist_hex_to_goal_y[i].item()}")
        #         print(f"  dist_hex_to_goal_z: {dist_hex_to_goal_z[i].item()}")
        #         print(f"  hex_pos_xyz: [{hex_pos_xyz[i, 0].item()}, {hex_pos_xyz[i, 1].item()}, {hex_pos_xyz[i, 2].item()}]")
        #         print(f"  hex_to_gripper_xyz:          [{states['hex_to_gripper_xyz'][i, 0].item()}, {states['hex_to_gripper_xyz'][i, 1].item()}, {states['hex_to_gripper_xyz'][i, 2].item()}]")

    else:
        hex_rewards = torch.zeros_like(cube_rewards)

    final_rewards = cube_rewards + hex_rewards

    # final_rewards = cube_rewards

    # for i in range(cube_placement_complete.shape[0]):
    #     cube_placement_progress = cube_y_align_flag[i].item() or (dist_cube_to_goal_z[i].item() and cube_lifted[i].item())
    #     # hex_placement_progress = dist_hex_to_hex_hole_xy[i].item() or (dist_hex_to_hex_hole_z[i].item() and hex_lifted[i].item())
    #     if cube_placement_progress: # or hex_placement_progress:
    #         print(f"Env {i}:")
    #         # cube_start_pose_xyz
    #         print("cube reward, hex reward", cube_rewards[i].item())

    #         print("CUBE INFO")
    #         print(f"  cube_placement_complete: [{cube_placement_complete[i].item()}]")
    #         print(f"  cube_lifted:         {cube_lifted[i].item()}")
    #         print(f"  dist_cube_to_goal_x: {dist_cube_to_goal_x[i].item()}")
    #         print(f"  dist_cube_to_goal_y: {dist_cube_to_goal_y[i].item()}")
    #         print(f"  dist_cube_to_goal_z: {dist_cube_to_goal_z[i].item()}")
    #         print(f"  gripper_released_cube: {gripper_released_cube[i].item()}")

    #         print(f"  cube_pos_xyz: [{cube_pos_xyz[i, 0].item()}, {cube_pos_xyz[i, 1].item()}, {cube_pos_xyz[i, 2].item()}]")
    #         print(f"  cube_to_gripper_xyz:       [{cube_to_gripper_xyz[i, 0].item()}, {cube_to_gripper_xyz[i, 1].item()}, {cube_to_gripper_xyz[i, 2].item()}]")
    #         print(f"  cube_hole_bottom_xyz:      [{cube_hole_bottom_xyz[i, 0].item()}, {cube_hole_bottom_xyz[i, 1].item()}, {cube_hole_bottom_xyz[i, 2].item()}]")
    #         print(f"  cube_to_gripper_xyz:          [{states['cube_to_gripper_xyz'][i, 0].item()}, {states['cube_to_gripper_xyz'][i, 1].item()}, {states['cube_to_gripper_xyz'][i, 2].item()}]")

    #         print(f"  cube_linear_velocity: [{cube_vel[i, 0].item()}, {cube_vel[i, 1].item()}, {cube_vel[i, 2].item()}]")
    #         print(f"  cube_angular_velocity:[{cube_ang_vel[i, 0].item()}, {cube_ang_vel[i, 1].item()}, {cube_ang_vel[i, 2].item()}]")
    #         print(f"   cube_align_reward: {cube_align_reward[i].item()}")
              
            # Print positions
            # print(f"Lifting Metrics || Is obj lifted:  cube: {cube_lifted[i].item()}")
            # print(f"  eef_lf_pos_xyz:            [{eef_lf_pos_xyz[i, 0].item()}, {eef_lf_pos_xyz[i, 1].item()}, {eef_lf_pos_xyz[i, 2].item()}]")
            # print(f"  eef_rf_pos_xyz:            [{eef_rf_pos_xyz[i, 0].item()}, {eef_rf_pos_xyz[i, 1].item()}, {eef_rf_pos_xyz[i, 2].item()}]")
            # print(f"  eef_state:                 [{eef_state[i, 0].item()}, {eef_state[i, 1].item()}, {eef_state[i, 2].item()}]")
            # print(f"  leff_finger_pos_xyz:           [{states['eef_lf_pos_xyz'][i, 0].item()}, {states['eef_lf_pos_xyz'][i, 1].item()}, {states['eef_lf_pos_xyz'][i, 2].item()}]")
            # print(f"  right_finger_pos_xyz:         [{states['eef_rf_pos_xyz'][i, 0].item()}, {states['eef_rf_pos_xyz'][i, 1].item()}, {states['eef_rf_pos_xyz'][i, 2].item()}]")
            # print(f"  gripper_pos_xyz:              [{states['eef_state'][i, 0].item()}, {states['eef_state'][i, 1].item()}, {states['eef_state'][i, 2].item()}]")

            # print(f"Alignment Metrics || Alignment Reward: cube: {cube_align_reward[i].item()}")
            # print(f"dist_cube_to_goal   {dist_cube_to_goal[i].item()}")
            # print(f"      cube_pos_xyz:                 [{states['cube_pos_xyz'][i, 0].item()}, {states['cube_pos_xyz'][i, 1].item()}, {states['cube_pos_xyz'][i, 2].item()}]")
            # print(f"      cube_goal_position_xyz:       [{cube_goal_position_xyz[i, 0].item()}, {cube_goal_position_xyz[i, 1].item()}, {cube_goal_position_xyz[i, 2].item()}]")
            # print(f"      cube_hole_bottom_xyz:       [{states['cube_hole_bottom_xyz'][i, 0].item()}, {states['cube_hole_bottom_xyz'][i, 1].item()}, {states['cube_hole_bottom_xyz'][i, 2].item()}]")
            # print(f"      cube_to_hole_bottom_xyz:    [{states['cube_to_hole_bottom_xyz'][i, 0].item()}, {states['cube_to_hole_bottom_xyz'][i, 1].item()}, {states['cube_to_hole_bottom_xyz'][i, 2].item()}]")
            # print(f"      dist_cube_to_cube_hole_xy:  {dist_cube_to_cube_hole_xy[i].item()}")
            # print(f"      dist_cube_to_cube_hole_z:   {dist_cube_to_cube_hole_z[i].item()} (cube_height_from_ground: {cube_height_from_ground[i].item()} vs target_cube_height: {target_cube_height[i].item()})")

            # print(f"      cube_linear_velocity:         [{states['cube_vel'][i, 0].item()}, {states['cube_vel'][i, 1].item()}, {states['cube_vel'][i, 2].item()}]")
            # print(f"      cube_angular_velocity:        [{states['cube_ang_vel'][i, 0].item()}, {states['cube_ang_vel'][i, 1].item()}, {states['cube_ang_vel'][i, 2].item()}]")
            
            # print("obj Placement Complete?: ", cube_placement_complete[i].item())
            # print("Reward Pieces")
            # print(f"      dist_reward cube:                  {scaled_dist_reward_cube[i].item()}")
            # print(f"      lift_reward cube:                  {scaled_lifted_reward_cube[i].item()}")
            # print(f"      alignment_reward cube:             {scaled_alignment_reward_cube[i].item()}")
            # print(f"      dist_reward hex:                  {scaled_dist_reward_hex[0].item()}")
            # print(f"      lift_reward hex:                  {scaled_lifted_reward_cube[0].item()}")
            # print(f"      alignment_reward hex:             {scaled_alignment_reward_cube[0].item()}")


    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    
    # WARNING: This will only work for a single environment right now
    # if torch.any(cube_placement_complete):

    #     if object_name not in objects_completed["objects"]:
    #         objects_completed["objects"].append(object_name)
        # new_element = torch.tensor([object_name], dtype=torch.float32)  # The element you want to append
        # empty_tensor = torch.cat((empty_tensor, new_element))
    # Evaluation
    # episode_success_counter = states["episode_success_counter"]
    # global_episode_counter = states["global_episode_counter"]
    # for i in range(progress_buf.shape[0]):
    #     # Check which environments are at the end of an episode
    #     if progress_buf[i] >= max_episode_length - 1:
    #         # Increment global episode counter for environments that are at the end
    #         global_episode_counter += 1
    #         if cube_placement_complete[i].item():
    #             # Update the success counter for environments where the cube_placement_complete is true at the end of an episode
    #             episode_success_counter[i] += 1


    # if progress_buf[0] >= max_episode_length - 1 and global_episode_counter % len(progress_buf) == 0:
    #     # Total episode successes
    #     total_episode_successes = episode_success_counter.sum().cpu().item()  # Move to CPU and convert to a Python scalar
    #     total_episodes = int(global_episode_counter.cpu().item())  # Ensure global_episode_counter is on CPU and convert to int
    #     failed_episodes = total_episodes - total_episode_successes

    #     print(f"Total Episodes: {total_episodes}")
    #     print(f"Successful Episodes: {total_episode_successes}")
    #     print(f"Failed Episodes: {failed_episodes}")

  
    return final_rewards, reset_buf, objects_completed