import numpy as np
import os
from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
import torch


class TestRobotReach(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka default end effector position (example for Franka Panda)
        self.robot_default_eef_pos = torch.tensor([-0.45, 0.0, 1.0], device=self.device)

        # Robot's maximum reach (for Franka, it is 0.85 meters)
        self.max_reach = 0.85

        # Target positions to test
        self.target_positions = [
            torch.tensor([0.6, 0.0, 1.0], device=self.device),  # Within reach
            torch.tensor([1.0, 0.0, 1.0], device=self.device),  # Outside reach
        ]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Test the reachability
        self.test_robot_reach()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0)

        self.frankas = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

    def test_robot_reach(self):
        for target_pos in self.target_positions:
            within_reach = self._is_within_reach(target_pos)
            if within_reach:
                print(f"Target position {target_pos.cpu().numpy()} is within reach.")
            else:
                print(f"Target position {target_pos.cpu().numpy()} is NOT within reach.")

    def _is_within_reach(self, target_position):
        """
        Check if a target position is within the robot's maximum reach.
        """
        current_eef_pos = self.robot_default_eef_pos
        distance = torch.norm(current_eef_pos - target_position).item()
        return distance <= self.max_reach


if __name__ == "__main__":
    # Placeholder configuration, should be updated as per your environment setup
    config = {
        "env": {
            "episodeLength": 1000,
            "actionScale": 1.0,
            "numObservations": 19,
            "numActions": 8,
            "envSpacing": 1.0,
        }
    }

    # Initialize the test environment
    test_env = TestRobotReach(config, rl_device="cuda:0", sim_device="cuda:0", graphics_device_id=0, headless=False, virtual_screen_capture=False, force_render=False)
