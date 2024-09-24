
from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask
import torch

def print_collision_info(gym, sim, actor_handle, env_ptr, name):
    # Get the collision group and filter for the specified actor
    collision_group = gym.get_actor_collision_group(env_ptr, actor_handle)
    filter_group = gym.get_actor_collision_filter(env_ptr, actor_handle)

    print(f"Actor: {name}")
    print(f"Collision Group: {collision_group}")
    print(f"Filter Group: {filter_group}")