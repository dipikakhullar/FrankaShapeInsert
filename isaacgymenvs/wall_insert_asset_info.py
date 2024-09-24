"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Asset and Environment Information
---------------------------------
Demonstrates introspection capabilities of the gym api at the asset and environment levels
- Once an asset is loaded its properties can be queried
- Assets in environments can be queried and their current states be retrieved
"""

import os
from isaacgym import gymapi
from isaacgym import gymutil


def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))


def print_actor_info(gym, env, actor_handle):

    name = gym.get_actor_name(env, actor_handle)

    body_names = gym.get_actor_rigid_body_names(env, actor_handle)
    body_dict = gym.get_actor_rigid_body_dict(env, actor_handle)

    joint_names = gym.get_actor_joint_names(env, actor_handle)
    joint_dict = gym.get_actor_joint_dict(env, actor_handle)

    dof_names = gym.get_actor_dof_names(env, actor_handle)
    dof_dict = gym.get_actor_dof_dict(env, actor_handle)

    print()
    print("===== Actor: %s =======================================" % name)

    print("\nBodies")
    print(body_names)
    print(body_dict)

    print("\nJoints")
    print(joint_names)
    print(joint_dict)

    print("\n Degrees Of Freedom (DOFs)")
    print(dof_names)
    print(dof_dict)
    print()

    # Get body state information
    body_states = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_ALL)

    # Print some state slices
    print("Poses from Body State:")
    print(body_states['pose'])          # print just the poses

    print("\nVelocities from Body State:")
    print(body_states['vel'])          # print just the velocities
    print()

    # iterate through bodies and print name and position
    body_positions = body_states['pose']['p']
    for i in range(len(body_names)):
        print("Body '%s' has position" % body_names[i], body_positions[i])

    print("\nDOF states:")

    # get DOF states
    dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

    # print some state slices
    # Print all states for each degree of freedom
    print(dof_states)
    print()

    # iterate through DOFs and print name and position
    dof_positions = dof_states['pos']
    for i in range(len(dof_names)):
        print("DOF '%s' has position" % dof_names[i], dof_positions[i])


# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Asset and Environment Information")

# create simulation context
sim_params = gymapi.SimParams()

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Print out the working directory
# helpful in determining the relative location that assets will be loaded from
print("Working directory: %s" % os.getcwd())

# Path where assets are searched, relative to the current working directory
# asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")        
asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")        

# List of assets that will be loaded, both URDF and MJCF files are supported
asset_files = ["urdf/franka_description/robots/franka_panda.urdf",
               "urdf/separate_wall_shapes_small/wall_pieces/small_hex_bottom.urdf",
               "urdf/separate_wall_shapes_small/wall_pieces/small_square_bottom.urdf",
               "urdf/separate_wall_shapes_small/wall_pieces/wall_top.urdf",
               "urdf/separate_wall_shapes_small/wall_pieces/small_triangle_bottom.urdf",
               "urdf/separate_wall_shapes_small/shape_objects/object_hex_05_15.urdf",
               "urdf/separate_wall_shapes_small/shape_objects/object_triangle_05_15.urdf",
               "urdf/separate_wall_shapes_small/shape_objects/object_cube_05_15.urdf",

               ]
asset_names = ["franka", "hex_bottom", "square_bottom", "wall_top", "triangle_bottom", "hex", "triangle", "cube"]
loaded_assets = []

# Load the assets and ensure that we are successful
for asset in asset_files:
    print("Loading asset '%s' from '%s'" % (asset, asset_root))

    current_asset = gym.load_asset(sim, asset_root, asset)

    if current_asset is None:
        print("*** Failed to load asset '%s'" % (asset, asset_root))
        quit()
    loaded_assets.append(current_asset)

for i in range(len(loaded_assets)):
    print()
    print_asset_info(loaded_assets[i], asset_names[i])

# Setup environment spacing
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

# Create one environment
env = gym.create_env(sim, lower, upper, 1)

# Add actors to environment
pose = gymapi.Transform()
for i in range(len(loaded_assets)):
    pose.p = gymapi.Vec3(0.0, 0.0, i * 2)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    gym.create_actor(env, loaded_assets[i], pose, asset_names[i], -1, -1)

print("=== Environment info: ================================================")

actor_count = gym.get_actor_count(env)
print("%d actors total" % actor_count)

# Iterate through all actors for the environment
for i in range(actor_count):
    actor_handle = gym.get_actor_handle(env, i)
    print_actor_info(gym, env, actor_handle)

# Cleanup the simulator
gym.destroy_sim(sim)













# print("IN CREATE ENV")

#         if "asset" in self.cfg["env"]:
#             asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
#             franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
#                # wall stuff 
#         self.table_height = 1
#         # Create table asset
#         table_pos = [0.0, 0.0, 1.0]
#         table_thickness = 0.05
#         table_opts = gymapi.AssetOptions()
#         table_opts.fix_base_link = True
#         table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)
#         table_start_pose = gymapi.Transform()
#         table_start_pose.p = gymapi.Vec3(*table_pos)
#         table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
#         self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
#         self.reward_settings["table_height"] = self._table_surface_pos[2]

#         # Create table stand asset
#         table_stand_height = 0.1
#         table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
#         table_stand_start_pose = gymapi.Transform()
#         table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
#         table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
#         table_stand_opts = gymapi.AssetOptions()
#         table_stand_opts.fix_base_link = True
#         table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

#         # load franka asset
#         asset_options = gymapi.AssetOptions()
#         asset_options.flip_visual_attachments = True
#         asset_options.fix_base_link = True
#         asset_options.collapse_fixed_joints = False
#         asset_options.disable_gravity = True
#         asset_options.thickness = 0.001
#         asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
#         asset_options.use_mesh_materials = True
#         franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
#         # Define start pose for franka
#         franka_start_pose = gymapi.Transform()
#         franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
#         franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
#         franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
#         franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

#         self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
#         self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

#         # set franka dof properties
#         franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
#         self.franka_dof_lower_limits = []
#         self.franka_dof_upper_limits = []
#         self._franka_effort_limits = []
#         for i in range(self.num_franka_dofs):
#             franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
#             if self.physics_engine == gymapi.SIM_PHYSX:
#                 franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
#                 franka_dof_props['damping'][i] = franka_dof_damping[i]
#             else:
#                 franka_dof_props['stiffness'][i] = 7000.0
#                 franka_dof_props['damping'][i] = 50.0

#             self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
#             self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
#             self._franka_effort_limits.append(franka_dof_props['effort'][i])

#         self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
#         self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
#         self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
#         self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
#         self.franka_dof_speed_scales[[7, 8]] = 0.1
#         franka_dof_props['effort'][7] = 200
#         franka_dof_props['effort'][8] = 200

#         shape_start_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), 1.5708)

#         # Create hex object asset
#         self.cube_height = 0.05
#         self.cube_length = 0.15
#         cube_opts = gymapi.AssetOptions()
#         # cube_opts.flip_visual_attachments = True       # Ensure correct visual attachment orientation
#         cube_opts.fix_base_link = False                # Allow the hexagon to move (not fixed to the base)
#         cube_opts.collapse_fixed_joints = False        # Maintain any joints in the asset structure if applicable
#         cube_opts.disable_gravity = False              # Enable gravity to make the object fall naturally unless controlled
#         cube_opts.thickness = 0.001                    # Specify mesh thickness if needed for accuracy
#         # cube_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS   # Enable position control for the gripper or joint actuation
#         cube_opts.use_mesh_materials = True            # Use materials from the mesh for accurate physics
#         cube_opts.vhacd_enabled = True                 # Enable convex hull decomposition for better collision handling
#         # cube_opts.vhacd_params = gymapi.VhacdParams()  # Use VHACD for detailed collision geometry
#         # cube_opts.vhacd_params.resolution = 1000000    # Adjust VHACD resolution for accurate collision
#         cube_asset = self.gym.load_asset(self.sim, asset_root, shape_asset_files_dict["cube_05_15"], cube_opts)
#         # cube_asset = self.gym.create_box(self.sim, *([self.cube_height, self.cube_height, self.cube_length]), cube_opts)
#         cube_color = gymapi.Vec3(1.0, 0.0, 0.0)
#         cube_start_pose = gymapi.Transform()
#         cube_start_pose.p = gymapi.Vec3(0, -0.15, self.table_height + table_thickness/2 + self.cube_height/2)
#         cube_start_pose.r = shape_start_quat

#         # Create hex object asset
#         self.hex_length = 0.15
#         hex_opts = gymapi.AssetOptions()
#         # hex_opts.flip_visual_attachments = True       # Ensure correct visual attachment orientation
#         hex_opts.fix_base_link = False                # Allow the hexagon to move (not fixed to the base)
#         hex_opts.collapse_fixed_joints = False        # Maintain any joints in the asset structure if applicable
#         hex_opts.disable_gravity = False              # Enable gravity to make the object fall naturally unless controlled
#         hex_opts.thickness = 0.001                    # Specify mesh thickness if needed for accuracy
#         # hex_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS   # Enable position control for the gripper or joint actuation
#         hex_opts.use_mesh_materials = True            # Use materials from the mesh for accurate physics
#         # hex_opts.vhacd_enabled = True                 # Enable convex hull decomposition for better collision handling
#         # hex_opts.vhacd_params = gymapi.VhacdParams()  # Use VHACD for detailed collision geometry
#         # hex_opts.vhacd_params.resolution = 1000000    # Adjust VHACD resolution for accurate collision
#         if self.hex_is_box > 0:
#             self.hex_height = 0.025
#             hex_asset = self.gym.create_box(self.sim, *[self.hex_height, self.hex_height, self.hex_length], hex_opts)
#         else:
#             self.hex_height = 0.05
#             hex_asset = self.gym.load_asset(self.sim, asset_root, shape_asset_files_dict["hex_05_15"], hex_opts)
#         # hex_asset = self.gym.create_box(self.sim, *([self.hex_height, self.hex_height, self.hex_length]), hex_opts)
#         hex_color = gymapi.Vec3(0.0, 0.4, 0.1)
#         hex_start_pose = gymapi.Transform()
#         hex_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_height + table_thickness/2 + self.hex_height/2)
#         hex_start_pose.r = shape_start_quat

#         # Create triangle object asset
#         self.triangle_height = 0.05
#         self.triangle_length = 0.15
#         triangle_opts = gymapi.AssetOptions()
#         triangle_opts.fix_base_link = False                # Allow the hexagon to move (not fixed to the base)
#         triangle_opts.collapse_fixed_joints = False        # Maintain any joints in the asset structure if applicable
#         triangle_opts.disable_gravity = False              # Enable gravity to make the object fall naturally unless controlled
#         triangle_opts.thickness = 0.001                    # Specify mesh thickness if needed for accuracy
#         # triangle_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS   # Enable position control for the gripper or joint actuation
#         triangle_opts.use_mesh_materials = True       
#         triangle_asset = self.gym.load_asset(self.sim, asset_root, shape_asset_files_dict["triangle_05_15"], triangle_opts)
#         triangle_color = gymapi.Vec3(1, 0.141, 0.922)
#         triangle_start_pose = gymapi.Transform()
#         triangle_start_pose.p = gymapi.Vec3(0.0, .15, self.table_height + table_thickness/2 + self.triangle_height/2)
#         triangle_start_pose.r = shape_start_quat


#         # Create Wall Asset
#         # Manually nter some of the wall dimensions
#         self._wall_dims_dict = {}
#         self._wall_dims_dict["wall_thickness"] = 0.25
#         self._wall_dims_dict["wall_height"] = 1.0
#         self._wall_dims_dict["wall_width"] = 0.5

#         # square_bottom
#         # center of square piece is right at the surface of the square shelf
#         # it is .3 high from the base of the square piece
#         # the total height of the square piece is .45
#         self._wall_dims_dict["square_bottom_z_offset_from_table"] = .3
#         self._wall_dims_dict["square_bottom_z_offset_from_hole"] = 0.0

#         # hex_bottom
#         # center of hex piece is at the surface of the hex shelf
#         # it is .1 aboe the base of the hex piece
#         # the total height of the hex piece is .25
#         self._wall_dims_dict["hex_bottom_z_offset_from_table"] = .45+.1
#         self._wall_dims_dict["hex_bottom_z_offset_from_hole"] = 0.0

#         # triangle_bottom
#         # center of the triangle piece is at the peak point of the triangle 
#         # it is .1 above the base of the triangle piece
#         # the total height of the triangle piece is .25
#         self._wall_dims_dict["triangle_bottom_z_offset_from_table"] = .45+.25+.1
#         self._wall_dims_dict["triangle_bottom_z_offset_from_hole"] = 0.0

#         # wall_top
#         # center of the wall top piece is at the center of the rectangle
#         # it is .075 above the base of the wall top piece
#         # the total height of the wall top piece is .15
#         self._wall_dims_dict["wall_top_z_offset_from_table"] = 0.45+.25+.25+.075
#         self._wall_dims_dict["wall_top_z_offset_from_hole"] = -0.075


#         wall_base_pose = [0.3, 0, self.table_height + table_thickness/2]


#         # 2. Custom Wall
#         # wall bottom square
#         square_bottom_wall_opts = gymapi.AssetOptions()
#         square_bottom_wall_opts.use_mesh_materials = True  
#         square_bottom_wall_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
#         square_bottom_wall_opts.override_com = True 
#         square_bottom_wall_opts.override_inertia = True 
#         square_bottom_wall_opts.disable_gravity = True
#         square_bottom_wall_opts.fix_base_link = True
#         square_bottom_wall_opts.vhacd_enabled = True 
#         square_bottom_wall_opts.vhacd_params = gymapi.VhacdParams() 
#         square_bottom_wall_opts.vhacd_params.resolution = 3000000
#         square_bottom_wall_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
#         square_bottom_start_xyz = [wall_base_pose[0], wall_base_pose[1], wall_base_pose[2]+self._wall_dims_dict["square_bottom_z_offset_from_table"]]
#         square_bottom_start_pos = gymapi.Transform()
#         square_bottom_start_pos.p = gymapi.Vec3(*square_bottom_start_xyz)
#         square_bottom_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
#         square_bottom_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['square_bottom'], square_bottom_wall_opts)


#         # hex bottom
#         hex_bottom_wall_opts = gymapi.AssetOptions()
#         hex_bottom_wall_opts.use_mesh_materials = True  
#         hex_bottom_wall_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
#         hex_bottom_wall_opts.override_com = True 
#         hex_bottom_wall_opts.override_inertia = True 
#         hex_bottom_wall_opts.disable_gravity = True
#         hex_bottom_wall_opts.fix_base_link = True
#         hex_bottom_wall_opts.vhacd_enabled = True 
#         hex_bottom_wall_opts.vhacd_params = gymapi.VhacdParams() 
#         hex_bottom_wall_opts.vhacd_params.resolution = 30000000
#         hex_bottom_wall_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
#         hex_bottom_start_xyz = [wall_base_pose[0], wall_base_pose[1], wall_base_pose[2]+self._wall_dims_dict["hex_bottom_z_offset_from_table"]]
#         # hex_bottom_start_xyz = [0,0,0]

#         hex_bottom_start_pos = gymapi.Transform()
#         hex_bottom_start_pos.p = gymapi.Vec3(*hex_bottom_start_xyz)
#         hex_bottom_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
#         hex_bottom_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['hex_bottom'], hex_bottom_wall_opts)

#         # triangle bottom

#         triangle_bottom_wall_opts = gymapi.AssetOptions()
#         triangle_bottom_wall_opts.use_mesh_materials = True  
#         triangle_bottom_wall_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
#         triangle_bottom_wall_opts.override_com = True 
#         triangle_bottom_wall_opts.override_inertia = True 
#         triangle_bottom_wall_opts.disable_gravity = True
#         triangle_bottom_wall_opts.fix_base_link = True
#         triangle_bottom_wall_opts.vhacd_enabled = True 
#         triangle_bottom_wall_opts.vhacd_params = gymapi.VhacdParams() 
#         triangle_bottom_wall_opts.vhacd_params.resolution = 30000000
#         triangle_bottom_wall_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
#         triangle_bottom_start_xyz = [wall_base_pose[0], wall_base_pose[1], wall_base_pose[2]+self._wall_dims_dict["triangle_bottom_z_offset_from_table"]]
#         # triangle_bottom_start_xyz = [0,0,0]

#         triangle_bottom_start_pos = gymapi.Transform()
#         triangle_bottom_start_pos.p = gymapi.Vec3(*triangle_bottom_start_xyz)
#         triangle_bottom_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
#         triangle_bottom_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['triangle_bottom'], triangle_bottom_wall_opts)


#         # wall top
#         wall_top_opts = gymapi.AssetOptions()
#         wall_top_opts.use_mesh_materials = True  
#         wall_top_opts.disable_gravity = True
#         wall_top_opts.fix_base_link = True
#         wall_top_start_xyz = [wall_base_pose[0], wall_base_pose[1], wall_base_pose[2]+self._wall_dims_dict["wall_top_z_offset_from_table"]]
#         # wall_top_start_xyz = [0,0,0]

#         wall_top_start_pos = gymapi.Transform()
#         wall_top_start_pos.p = gymapi.Vec3(*wall_top_start_xyz)
#         wall_top_start_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0)
#         wall_top_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_files_dict['wall_top'], wall_top_opts)

#         # count franka asset nums
#         num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
#         num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)

#         # count wall asset nums
#         square_bottom_bodies = self.gym.get_asset_rigid_body_count(square_bottom_asset)
#         square_bottom_shapes = self.gym.get_asset_rigid_shape_count(square_bottom_asset)

#         hex_bottom_bodies = self.gym.get_asset_rigid_body_count(hex_bottom_asset)
#         hex_bottom_shapes = self.gym.get_asset_rigid_shape_count(hex_bottom_asset)

#         triangle_bottom_bodies = self.gym.get_asset_rigid_body_count(triangle_bottom_asset)
#         triangle_bottom_shapes = self.gym.get_asset_rigid_shape_count(triangle_bottom_asset)

#         wall_top_bodies = self.gym.get_asset_rigid_body_count(wall_top_asset)
#         wall_top_shapes = self.gym.get_asset_rigid_shape_count(wall_top_asset)

#         # count shape object asset nums
#         cube_bodies = self.gym.get_asset_rigid_body_count(cube_asset)
#         cube_shapes = self.gym.get_asset_rigid_shape_count(cube_asset)

#         hex_bodies = self.gym.get_asset_rigid_body_count(hex_asset)
#         hex_shapes = self.gym.get_asset_rigid_shape_count(hex_asset)
        
#         triangle_bodies = self.gym.get_asset_rigid_body_count(triangle_asset)
#         triangle_shapes = self.gym.get_asset_rigid_shape_count(triangle_asset)

#         # count table asset nums
#         num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
#         num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        
#         num_table_stand_bodies = self.gym.get_asset_rigid_body_count(table_stand_asset)
#         num_table_stand_shapes = self.gym.get_asset_rigid_shape_count(table_stand_asset)


#         max_agg_bodies = num_franka_bodies + square_bottom_bodies + hex_bottom_bodies \
#                         + triangle_bottom_bodies + wall_top_bodies + num_table_bodies \
#                         +  num_table_stand_bodies + cube_bodies  + hex_bodies \
#                         + triangle_bodies
#         max_agg_shapes = num_franka_shapes + square_bottom_shapes + hex_bottom_shapes \
#                         + triangle_bottom_shapes + wall_top_shapes + num_table_shapes \
#                         + num_table_stand_shapes + cube_shapes + hex_shapes \
#                         + triangle_shapes

#         # Print the number of rigid bodies and shapes for each asset
#         print(f"Franka: Bodies: {num_franka_bodies}, Shapes: {num_franka_shapes}")
#         print(f"Square Bottom: Bodies: {square_bottom_bodies}, Shapes: {square_bottom_shapes}")
#         print(f"Hex Bottom: Bodies: {hex_bottom_bodies}, Shapes: {hex_bottom_shapes}")
#         print(f"Wall Top: Bodies: {wall_top_bodies}, Shapes: {wall_top_shapes}")

#         print(f"Table: Bodies: {num_table_bodies}, Shapes: {num_table_shapes}")
#         print(f"Table Stand: Bodies: {num_table_stand_bodies}, Shapes: {num_table_stand_shapes}")
#         print(f"obj A: Bodies: {cube_bodies}, Shapes: {cube_shapes}")
#         print(f"Hexagon: Bodies: {hex_bodies}, Shapes: {hex_shapes}")
#         print(f"Triangle: Bodies: {triangle_bodies}, Shapes: {triangle_shapes}")


#         self.frankas = []
#         self.envs = []
#         for i in range(self.num_envs):
#             # create env instance

#             # print(self.aggregate_mode)
#             env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

#             # Create actors and define aggregate group appropriately depending on setting
#             # NOTE: franka should ALWAYS be loaded first in sim!
#             if self.aggregate_mode >= 3:
#                 self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

#             # Create franka
#             # Potentially randomize start pose
#             if self.franka_position_noise > 0:
#                 rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
#                 franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
#                                                  1.0 + table_thickness / 2 + table_stand_height)
#             if self.franka_rotation_noise > 0:
#                 rand_rot = torch.zeros(1, 3)
#                 rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
#                 new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
#                 franka_start_pose.r = gymapi.Quat(*new_quat)
            
#             # 1. Franka
#             # print("franka", self.aggregate_mode)
#             franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
#             self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

#             if self.aggregate_mode == 2:
#                 self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

#             # 2. Custom Wall
#             self._square_bottom_id = self.gym.create_actor(env_ptr, square_bottom_asset, square_bottom_start_pos, "square_bottom", i, 1, 0)

#             # hex bottom
#             self._hex_bottom_id = self.gym.create_actor(env_ptr, hex_bottom_asset, hex_bottom_start_pos, "hex_bottom", i, 1, 0)

#             # hex bottom
#             self._triangle_bottom_id = self.gym.create_actor(env_ptr, triangle_bottom_asset, triangle_bottom_start_pos, "triangle_bottom", i, 1, 0)

#             # wall top
#             self._wall_top_id = self.gym.create_actor(env_ptr, wall_top_asset, wall_top_start_pos, "wall_top", i, 1, 0)

#             # 3. Table
#             self._table_id = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
#             self._table_stand_id = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0)

#             if self.aggregate_mode == 1:
#                 self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

#             # 4. Create Objects
#             # make sure you create obj last because something is hardcoded in a reset
#             self._triangle_id = self.gym.create_actor(env_ptr, triangle_asset, triangle_start_pose, "triangle", i, 5, 0)
#             self.gym.set_rigid_body_color(env_ptr, self._triangle_id, 0, gymapi.MESH_VISUAL, triangle_color)
#             triangle_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._triangle_id)[0]
            
#             self._hex_id = self.gym.create_actor(env_ptr, hex_asset, hex_start_pose, "hex", i, 4, 0)
#             self.gym.set_rigid_body_color(env_ptr, self._hex_id, 0, gymapi.MESH_VISUAL, hex_color)
#             hex_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._hex_id)[0]
#             # print("hex mass", hex_props.mass)

#             self._cube_id = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 4, 0)
#             self.gym.set_rigid_body_color(env_ptr, self._cube_id, 0, gymapi.MESH_VISUAL, cube_color)
#             cube_props = self.gym.get_actor_rigid_body_properties(env_ptr, self._cube_id)[0]
#             # print("obj mass", props.mass)


#             if self.aggregate_mode > 0:
#                 self.gym.end_aggregate(env_ptr)

            
#             # Store the created env pointers
#             self.envs.append(env_ptr)
#             # print("SELF.ENVS", self.envs)
#             self.frankas.append(franka_actor)

#         # Setup init state buffer
#         self._init_cube_state = torch.zeros(self.num_envs, 13, device=self.device)
#         self._init_hex_state = torch.zeros(self.num_envs, 13, device=self.device)
#         self._init_triangle_state = torch.zeros(self.num_envs, 13, device=self.device)

#         obj_goal_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 1.5708)
#         self._desired_obj_quat = torch.zeros(self.num_envs, 4, device=self.device)
#         self._desired_obj_quat[:, 0] = obj_goal_quat.x
#         self._desired_obj_quat[:, 1] = obj_goal_quat.y
#         self._desired_obj_quat[:, 2] = obj_goal_quat.z
#         self._desired_obj_quat[:, 3] = obj_goal_quat.w