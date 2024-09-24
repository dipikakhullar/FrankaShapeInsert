# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .ant import Ant
from .anymal import Anymal
from .anymal_terrain import AnymalTerrain
from .ball_balance import BallBalance
from .cartpole import Cartpole 
from .factory.factory_task_gears import FactoryTaskGears
from .factory.factory_task_insertion import FactoryTaskInsertion
from .factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
from .factory.factory_task_nut_bolt_place import FactoryTaskNutBoltPlace
from .factory.factory_task_nut_bolt_screw import FactoryTaskNutBoltScrew
from .franka_cabinet import FrankaCabinet
from .franka_cube_stack import FrankaCubeStack
from .humanoid import Humanoid
# from .humanoid_amp import HumanoidAMP
from .ingenuity import Ingenuity
from .quadcopter import Quadcopter
from .shadow_hand import ShadowHand
from .allegro_hand import AllegroHand
from .dextreme.allegro_hand_dextreme import AllegroHandDextremeManualDR, AllegroHandDextremeADR
from .trifinger import Trifinger

from .allegro_kuka.allegro_kuka_reorientation import AllegroKukaReorientation
from .allegro_kuka.allegro_kuka_regrasping import AllegroKukaRegrasping
from .allegro_kuka.allegro_kuka_throw import AllegroKukaThrow
from .allegro_kuka.allegro_kuka_two_arms_regrasping import AllegroKukaTwoArmsRegrasping
from .allegro_kuka.allegro_kuka_two_arms_reorientation import AllegroKukaTwoArmsReorientation

from .industreal.industreal_task_pegs_insert import IndustRealTaskPegsInsert
from .industreal.industreal_task_gears_insert import IndustRealTaskGearsInsert




# Import your new task
from isaacgymenvs.tasks.cube_cubby_task import CubeCubbyTask
from isaacgymenvs.tasks.cube_cubby_task2 import CubeCubbyTask2
from isaacgymenvs.tasks.cube_cubby_task_custom_wall import CubeCubbyTaskCustomWall
from isaacgymenvs.tasks.cube_cubby_task_custom_wall_sc import CubeCubbyTaskCustomWallSC
from isaacgymenvs.tasks.generalized_wall_insert import GeneralizedWallInsert
from isaacgymenvs.tasks.split_shape_wall_insert import SplitShapeWallInsert
from isaacgymenvs.tasks.split_shape_wall_insert_small_holes import SplitShapeWallInsertSmallHoles
from isaacgymenvs.tasks.split_shape_wall_insert_small_holes2s import SplitShapeWallInsertSmallHoles2S
from isaacgymenvs.tasks.trial_1 import Trial1
from isaacgymenvs.tasks.wall_insert2s import WallInsert2S
from isaacgymenvs.tasks.generalized_wall_insert_horizontal import GeneralizedWallInsertHorizontal

def resolve_allegro_kuka(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        reorientation=AllegroKukaReorientation,
        throw=AllegroKukaThrow,
        regrasping=AllegroKukaRegrasping,
    )

    if subtask_name not in subtask_map:
        print("!!!!!")
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)

def resolve_allegro_kuka_two_arms(cfg, *args, **kwargs):
    subtask_name: str = cfg["env"]["subtask"]
    subtask_map = dict(
        reorientation=AllegroKukaTwoArmsReorientation,
        regrasping=AllegroKukaTwoArmsRegrasping,
    )

    if subtask_name not in subtask_map:
        raise ValueError(f"Unknown subtask={subtask_name} in {subtask_map}")

    return subtask_map[subtask_name](cfg, *args, **kwargs)


# Mappings from strings to environments
isaacgym_task_map = {
    "AllegroHand": AllegroHand,
    "AllegroKuka": resolve_allegro_kuka,
    "AllegroKukaTwoArms": resolve_allegro_kuka_two_arms,
    "AllegroHandManualDR": AllegroHandDextremeManualDR,
    "AllegroHandADR": AllegroHandDextremeADR,
    "Ant": Ant,
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "BallBalance": BallBalance,
    "Cartpole": Cartpole,
    "FactoryTaskGears": FactoryTaskGears,
    "FactoryTaskInsertion": FactoryTaskInsertion,
    "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
    "FactoryTaskNutBoltPlace": FactoryTaskNutBoltPlace,
    "FactoryTaskNutBoltScrew": FactoryTaskNutBoltScrew,
    "IndustRealTaskPegsInsert": IndustRealTaskPegsInsert,
    "IndustRealTaskGearsInsert": IndustRealTaskGearsInsert,
    "FrankaCabinet": FrankaCabinet,
    "FrankaCubeStack": FrankaCubeStack,
    "Humanoid": Humanoid,
    # "HumanoidAMP": HumanoidAMP,
    "Ingenuity": Ingenuity,
    "Quadcopter": Quadcopter,
    "ShadowHand": ShadowHand,
    "Trifinger": Trifinger,
    'CubeCubbyTask': CubeCubbyTask,  
    'CubeCubbyTask2': CubeCubbyTask2,

    'Trial1': Trial1, # Register your task here,
    'SplitShapeWallInsert': SplitShapeWallInsert,
    'GeneralizedWallInsert': GeneralizedWallInsert,
    'SplitShapeWallInsertSmallHoles': SplitShapeWallInsertSmallHoles,
    'SplitShapeWallInsertSmallHoles2S': SplitShapeWallInsertSmallHoles2S,
    'CubeCubbyTaskCustomWall': CubeCubbyTaskCustomWall,
    'CubeCubbyTaskCustomWallSC': CubeCubbyTaskCustomWallSC,
    "WallInsert2S": WallInsert2S,
    "GeneralizedWallInsertHorizontal": GeneralizedWallInsertHorizontal,

}
