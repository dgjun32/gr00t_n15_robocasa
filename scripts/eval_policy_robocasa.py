# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import json
import os
import warnings
from collections import defaultdict
from glob import glob
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robocasa
import robosuite
from gymnasium.wrappers import TimeLimit
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy

warnings.simplefilter("ignore", category=FutureWarning)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="host")
    parser.add_argument("--port", type=int, default=5555, help="port")
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr1_arms_only",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--video_backend", type=str, default="decord")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace/")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    ## When using a model instead of client-server mode.
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )

    # robocasa env and evaluation parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the robocasa environment",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CloseDrawer",
        help="Name of the robocasa environment to load",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save the video",
    )

    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument(
        "--layout_and_style_ids",
        type=int,
        nargs="+",
        default=[1, 1, 2, 2, 4, 4, 6, 9, 7, 10],
        help=(
            "Flattened pairs of (layout_id, style_id): l1 s1 l2 s2 ... (length must be even). "
            "Default: (1,1) (2,2) (4,4) (6,9) (7,10)."
        ),
    )
    parser.add_argument("--generative_textures", action="store_true", help="Use generative textures")

    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    all_gt_actions = []
    all_pred_actions = []

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # Parse flattened layout_and_style_ids into list of (layout, style) pairs
    flat_ls = args.layout_and_style_ids
    assert len(flat_ls) % 2 == 0, "--layout_and_style_ids must contain an even number of integers (layout style ...)"
    layout_and_style_pairs = [(int(flat_ls[i]), int(flat_ls[i + 1])) for i in range(0, len(flat_ls), 2)]

    env = load_robocasa_gym_env(
        args.env_name,
        seed=args.seed,
        generative_textures="100p" if args.generative_textures else None,
        layout_and_style_ids=layout_and_style_pairs,
        layout_ids=None,
        style_ids=None,
    )
    env = RoboCasaWrapper(env)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
    record_video = args.video_path is not None
    if record_video:
        video_base_path = Path(args.video_path)
        video_base_path.mkdir(parents=True, exist_ok=True)
        print(f"Video will be saved to: {video_base_path.absolute()}")
        # Record every episode (episode_id starts from 1)
        episode_trigger = lambda t: True  # Record all episodes
        try:
            env = RecordVideo(env, video_base_path, disable_logger=True, episode_trigger=episode_trigger, fps=20)
            print("RecordVideo wrapper added successfully")
        except Exception as e:
            print(f"WARNING: Failed to add RecordVideo wrapper: {e}")
            print("Continuing without video recording...")
            record_video = False

    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=args.action_horizon,
    )

    # postprocess function of action, to handle the case where number of dimensions are not the same
    def postprocess_action(action):
        new_action = {}
        for k, v in action.items():
            if v.ndim == 1:
                new_action[k] = v[..., None]
            else:
                new_action[k] = v
        return new_action


    # main evaluation loop
    stats = defaultdict(list)
    
    # Helper function to sanitize filename
    def sanitize_filename(name):
        """Remove or replace characters that are invalid in filenames."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        # Also replace spaces and limit length
        name = name.replace(' ', '_')
        return name[:100]  # Limit length to avoid too long filenames
    
    # Store episode info for renaming videos later
    episode_info_list = []
    
    for i in trange(args.num_episodes):
        print(f"\n{'='*80}")
        print(f"Starting Episode {i+1}/{args.num_episodes}")
        print(f"{'='*80}")
        
        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"ERROR during env.reset() in episode {i+1}: {e}")
            import traceback
            traceback.print_exc()
            break
        # get initial pose of the robot ############
        unwrapped_env = env.unwrapped
        sim = unwrapped_env.sim
        robot = unwrapped_env.robots[0]
        initial_robot_qpos = {}
        initial_robot_qvel = {}
        # Use robot_joints instead of joints
        for joint in robot.robot_joints:
            joint_id = sim.model.joint_name2id(joint)
            qpos_addr = sim.model.jnt_qposadr[joint_id]
            qvel_addr = sim.model.jnt_dofadr[joint_id]
            initial_robot_qpos[joint] = sim.data.qpos[qpos_addr].copy()
            initial_robot_qvel[joint] = sim.data.qvel[qvel_addr].copy()
        ############################################
        task_instruction = env.unwrapped.get_ep_meta()['lang']
        print(f"Task instruction: {task_instruction}")
        pbar = tqdm(
            total=args.max_episode_steps, desc=f"Episode {i + 1} / {env.unwrapped.get_ep_meta()['lang']}", leave=False
        )
        done = False
        step = 0
        while not done:
            def summarize_np(x):
                return dict(
                    shape=getattr(x, "shape", None),
                    dtype=getattr(x, "dtype", None),
                    any_nan=np.isnan(x).any() if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number) else False,
                    any_inf=np.isinf(x).any() if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number) else False,
                    min=(np.nanmin(x) if isinstance(x, np.ndarray) and x.size and np.issubdtype(x.dtype, np.number) else None),
                    max=(np.nanmax(x) if isinstance(x, np.ndarray) and x.size and np.issubdtype(x.dtype, np.number) else None),
                )

            # print("=== OBS SUMMARY ===")
            # for k, v in obs.items():
            #     print(k, summarize_np(v))
            action = policy.get_action(obs)
            # print("updated action", action)

            post_action = postprocess_action(action)
            next_obs, reward, terminated, truncated, info = env.step(post_action)

            done = terminated or truncated
            step += args.action_horizon
            obs = next_obs
            pbar.update(args.action_horizon)
            
            if step >= args.max_episode_steps:
                break
        add_to(stats, flatten({"is_success": info["is_success"]}))
        
        # Calculate cumulative success rate
        total_episodes = i + 1
        total_successes = int(sum(stats["is_success"]))
        success_rate = total_successes / total_episodes
        is_success_val = bool(info["is_success"])
        
        # Store episode info for later video renaming (videos are saved after env.close())
        if record_video:
            episode_info_list.append({
                "episode_idx": i + 1,
                "task_instruction": task_instruction,
                "is_success": is_success_val,
            })
        print(f"Episode {i+1}: Success = {is_success_val} | Success rate = {success_rate:.3f}")
        
        pbar.close()
        
        # Force garbage collection after each episode to prevent memory buildup
        import gc
        gc.collect()
        
        # Optional: clear CUDA cache if using GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    # Close environment to ensure video is saved
    print(f"\nClosing environment after {len(stats['is_success'])} episodes...")
    try:
        env.close()
        print("Environment closed successfully")
    except Exception as e:
        print(f"Error closing environment: {e}")
        import traceback
        traceback.print_exc()

    # Rename videos with custom format after all videos are saved
    if record_video:
        print("\n" + "="*80)
        print("Renaming video files...")
        print("="*80)
        
        renamed_count = 0
        for ep_info in episode_info_list:
            episode_idx = ep_info["episode_idx"]
            task_instruction = ep_info["task_instruction"]
            is_success = ep_info["is_success"]
            
            # Original filename created by RecordVideo
            old_filename = f"rl-video-episode-{episode_idx}.mp4"
            old_video_path = video_base_path / old_filename
            
            if old_video_path.exists():
                # Create new filename
                success_label = "success" if is_success else "fail"
                task_clean = sanitize_filename(task_instruction)
                new_filename = f"{success_label}_{task_clean}_{episode_idx}.mp4"
                new_video_path = video_base_path / new_filename
                
                # Rename the file
                old_video_path.rename(new_video_path)
                print(f"✓ Episode {episode_idx}: {old_filename} → {new_filename}")
                renamed_count += 1
            else:
                print(f"✗ Episode {episode_idx}: Video file not found: {old_filename}")
        
        print(f"\nRenamed {renamed_count}/{len(episode_info_list)} video files")
        
        # List all final video files
        video_files = list(video_base_path.glob("*.mp4"))
        print(f"\nFinal video files: {len(video_files)}")
        for video_file in sorted(video_files):
            print(f"  - {video_file.name}")
        
        if len(video_files) == 0:
            print("⚠️  No video files were saved. Check the render method and RecordVideo wrapper.")

    for k, v in stats.items():
        stats[k] = np.mean(v)
    print(stats)

    exit()
