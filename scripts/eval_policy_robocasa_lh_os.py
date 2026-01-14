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
import cv2
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import shutil

import cv2
import h5py
import mujoco
import numpy as np
import robocasa
import robosuite
from google import genai
from google.genai import types
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



def augment_subgoal_in_video(video_path, subgoal_list, output_path=None):
    """
    Augment video with subgoal text overlays at specified time intervals.
    
    Args:
        video_path (str): Path to input video file
        subgoal_list (List): Dictionary mapping subgoal text to [start_frame, end_frame]
                            e.g., [("subgoal 1", [0, 200]), ("subgoal 2", [200, 400]), ...]
        output_path (str, optional): Path to save output video. If None, saves as {video_path}_annotated.mp4
    
    Returns:
        str: Path to the output video file
    """
    video_path = Path(video_path)
    
    # Set output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_annotated{video_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer - try multiple codecs for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("⚠️ Warning: VideoWriter failed to open, trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.with_suffix('.avi')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
    if not out.isOpened():
        raise ValueError("Failed to create video writer with any codec")
    
    # Create frame-to-subgoal mapping for faster lookup
    frame_to_subgoal = {}
    for subgoal_text, (start_frame, end_frame) in subgoal_list:
        for frame_idx in range(start_frame, min(end_frame, total_frames)):
            frame_to_subgoal[frame_idx] = subgoal_text
    
    print(f"Created subgoal mapping for {len(frame_to_subgoal)} frames")
    
    # Process each frame
    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add subgoal text if exists for this frame
            if frame_idx in frame_to_subgoal:
                subgoal_text = frame_to_subgoal[frame_idx]
                
                # Text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6  # Smaller font for 240x240 resolution
                font_thickness = 2
                text_color = (0, 0, 255)  # Red (BGR format)
                bg_color = (0, 0, 0)  # Black background
                padding = 10
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    subgoal_text, font, font_scale, font_thickness
                )
                
                # Calculate position (top center)
                text_x = (width - text_width) // 2
                text_y = padding + text_height
                
                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (text_x - padding, text_y - text_height - padding),
                    (text_x + text_width + padding, text_y + baseline + padding),
                    bg_color,
                    -1  # Filled rectangle
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    subgoal_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    # Re-encode with ffmpeg for better compatibility
    temp_path = output_path
    final_path = output_path.parent / f"{output_path.stem}_final{output_path.suffix}"
    
    print(f"Re-encoding with ffmpeg for better compatibility...")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_path),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p', str(final_path)
        ], check=True, capture_output=True)
        
        # Replace temp file with final file
        shutil.move(str(final_path), str(temp_path))
        print(f"✅ Annotated video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ ffmpeg re-encoding failed, using original output")
        print(f"✅ Annotated video saved to: {output_path}")
    except FileNotFoundError:
        print(f"⚠️ ffmpeg not found, using original output")
        print(f"✅ Annotated video saved to: {output_path}")
    
    return str(output_path)


from PIL import Image
import io
import time
PLANNING_PROMPT = """
You are an agent that plans subgoals for performing a robotic task.
Each subgoal you plan will be executed by a low-level Vision-Language-Action (VLA) policy that converts it into motor actions.
You are given three types of inputs:
- Global task: the high-level task instruction provided by a human.
- Previous observation: the visual state from the previous timestamp (first image).
- Previous subtask: the subgoal that was planned based on the previous observation.
- Current observation: the visual state after executing the previous subtask (second image).

Using the above information, reason whether the previous subtask was successfully executed.
Then, based on that reasoning, propose the next subtask that should be executed to accomplish the global task.
The subtask should specify the object to be interacted with and the arm to be used.

Here are the inputs:
Global task: {global_task}
Previous subtask: {prev_subtask}

Your output should follow the following format:
<think>
In this block, describe the reasoning process 
(e.g., tracking the progress of the previous subtask, describing the current state, inferring next subgoal to achieve the global task).
</think>
<next_subgoal>
Describe the next subtask that should be executed to accomplish the global task.
</next_subgoal>
"""


PLANNING_PROMPT_CURR_MULTIVIEW = """
You are an agent that plans subgoals for performing a robotic task.
Each subgoal you plan will be executed by a low-level Vision-Language-Action (VLA) policy that converts it into motor actions.
You are given three types of inputs:
- Global task: the high-level task instruction provided by a human.
- Previous subtask: the subgoal that was planned based on the previous observation.
- Current observations: the visual states from the current timestamp.
    - First image: the right view of the robot.
    - Second image: the left view of the robot.
    - Third image: the wrist view of the robot.

Using the above information, reason whether the previous subtask was successfully executed.
Then, based on that reasoning, propose the next subtask that should be executed to accomplish the global task.
The subtask should specify the object to be interacted with.

Here are the inputs:
Global task: {global_task}
Previous subtask: {prev_subtask}

Your output should follow the following format:
<think>
In this block, describe the reasoning process 
(e.g., tracking the progress of the previous subtask, describing the current state, inferring next subgoal to achieve the global task).
</think>
<next_subgoal>
Describe the next subtask that should be executed to accomplish the global task.
</next_subgoal>
"""

PLANNING_PROMPT_ONESHOT = """
You are an agent that generates a plan for performing a following robotic task: {global_task}
You are given two types of inputs:
- Global task: the high-level task instruction provided by a human.
- Current observation: the visual state from the current timestamp.
 - First image: the right view of the robot.
 - Second image: the left view of the robot.
 - Third image: the wrist view of the robot.

Using the above information, provide a plan for the entire task.

Your output should follow the following format:
<think>
In this block, describe the reasoning process 
(e.g., tracking the progress of the previous subtask, describing the current state, inferring next subgoal to achieve the global task).
</think>
<plan>
[
    {{
        "subgoal": "...",
        "time_budget": "..."
    }},
    {{
        "subgoal": "...",
        "time_budget": "..."
    }},
    ...
    {{
        "subgoal": "...",
        "time_budget": "..."
    }}
]
</plan>

Ensure the following constraints:
    - Note that you should complete the entire task within the time budget of 3000 steps. 
      Therefore, the total time budget of the plan should be less than 3000 steps.
      Distribute the time budget according to the complexity of each subgoal.
    - Note that each subgoal should one of the following types:
      * pick up an object and place it in a specific location
      * open / close a drawer, door, microwave, etc.
      * press a button of a microwave, coffee machine, etc.
"""

class HighLevelPlanner:
    def __init__(self, model_name: str, max_retries: int = 5, retry_delay: int = 2):
        self.client = genai.Client(api_key='AIzaSyDpPcXVrt_dcRRqfGbJ0KOWIvmlT6zWwjw') #os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def numpy_to_bytes(self, img_array):
        """Convert numpy array to PNG bytes"""
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def get_subgoal_instruction(self, obs: dict, prev_subgoal: str, prev_image: np.ndarray) -> str:
        high_level_instruction = obs['annotation.human.action.task_description']
        curr_image = obs['video.right_view'].squeeze(0)
        
        prev_image_bytes = self.numpy_to_bytes(prev_image)
        curr_image_bytes = self.numpy_to_bytes(curr_image)
        
        contents = [
            types.Part.from_bytes(data=prev_image_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=curr_image_bytes, mime_type="image/png"),
            PLANNING_PROMPT.format(global_task=high_level_instruction, prev_subtask=prev_subgoal)
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents = contents,
                    config = types.GenerateContentConfig(
                        temperature=0.5,
                        thinking_config=types.ThinkingConfig(thinking_budget=-1)
                    )
                )
                output = response.text
                think = output.split("<think>")[1].split("</think>")[0]
                next_subgoal = output.split("<next_subgoal>")[1].split("</next_subgoal>")[0]
                return think, next_subgoal
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e
    
    def get_subgoal_instruction_curr_multiview(self, obs: dict, prev_subgoal: str) -> str:
        high_level_instruction = obs['annotation.human.action.task_description']
        right_image = obs['video.right_view'].squeeze(0)
        left_image = obs['video.left_view'].squeeze(0)
        wrist_image = obs['video.wrist_view'].squeeze(0)
        
        right_image_bytes = self.numpy_to_bytes(right_image)
        left_image_bytes = self.numpy_to_bytes(left_image)
        wrist_image_bytes = self.numpy_to_bytes(wrist_image)
        
        contents = [
            types.Part.from_bytes(data=right_image_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=left_image_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=wrist_image_bytes, mime_type="image/png"),
            PLANNING_PROMPT_CURR_MULTIVIEW.format(global_task=high_level_instruction, prev_subtask=prev_subgoal)
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents = contents,
                    config = types.GenerateContentConfig(
                        temperature=0.5,
                        thinking_config=types.ThinkingConfig(thinking_budget=-1)
                    )
                )
                output = response.text
                think = output.split("<think>")[1].split("</think>")[0]
                next_subgoal = output.split("<next_subgoal>")[1].split("</next_subgoal>")[0]
                return think, next_subgoal
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e
                
    def get_oneshot_plan(self, obs: dict, global_task: str) -> str:
        right_image = obs['video.right_view'].squeeze(0)
        left_image = obs['video.left_view'].squeeze(0)
        wrist_image = obs['video.wrist_view'].squeeze(0)
        
        right_image_bytes = self.numpy_to_bytes(right_image)
        left_image_bytes = self.numpy_to_bytes(left_image)
        wrist_image_bytes = self.numpy_to_bytes(wrist_image)
        
        contents = [
            types.Part.from_bytes(data=right_image_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=left_image_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=wrist_image_bytes, mime_type="image/png"),
            PLANNING_PROMPT_ONESHOT.format(global_task=global_task)
        ]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents = contents,
                    config = types.GenerateContentConfig(
                        temperature=0.5,
                        thinking_config=types.ThinkingConfig(thinking_budget=-1)
                    )
                )
                output = response.text
                think = output.split("<think>")[1].split("</think>")[0]
                plan = output.split("<plan>")[1].split("</plan>")[0]
                return think, plan
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise e
                

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
        "--planner_model_name",
        type=str,
        default=None,
        help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",
    )
    
    parser.add_argument(
        "--subgoal_interval",
        type=int,
        default=100,
        help="Interval between subgoal instructions.",
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
        
    # initialize high-level planner
    planner = HighLevelPlanner(args.planner_model_name)

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
        env = RecordVideo(env, video_base_path, disable_logger=True, episode_trigger=episode_trigger, fps=20)
        print("RecordVideo wrapper added successfully")

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
    subgoal_info_list = []
    
    for i in trange(args.num_episodes):

        obs, info = env.reset()
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
        initial_robot_state = {'qpos': initial_robot_qpos, 'qvel': initial_robot_qvel}
        ############################################
        
        task_instruction = env.unwrapped.get_ep_meta()['lang']
        think, plan = planner.get_oneshot_plan(obs, task_instruction)
        plan_list = json.loads(plan)
        print(f"Task instruction: {task_instruction}")
        print(f"Think: {think}")
        print(f"Plan: {plan}")
        pbar = tqdm(
            total=args.max_episode_steps, desc=f"Episode {i + 1}", leave=False
        )
        done = False
        step = 0
        prev_subgoal = "No previous subgoal provided yet."
        episode_subgoal_info = []
        for subtask in plan_list:
            subgoal_instruction = subtask['subgoal']
            time_budget = subtask['time_budget']
            def summarize_np(x):
                return dict(
                    shape=getattr(x, "shape", None),
                    dtype=getattr(x, "dtype", None),
                    any_nan=np.isnan(x).any() if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number) else False,
                    any_inf=np.isinf(x).any() if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number) else False,
                    min=(np.nanmin(x) if isinstance(x, np.ndarray) and x.size and np.issubdtype(x.dtype, np.number) else None),
                    max=(np.nanmax(x) if isinstance(x, np.ndarray) and x.size and np.issubdtype(x.dtype, np.number) else None),
                )
            subtask_step = 0
            # for every K steps, call high-level policy to generate subgoal instruction
            while subtask_step < int(time_budget):
                obs['annotation.human.action.task_description'] = [subgoal_instruction]
                action = policy.get_action(obs)
                # print("updated action", action)

                post_action = postprocess_action(action)
                next_obs, reward, terminated, truncated, info = env.step(post_action)
                
                done = terminated or truncated
                subtask_step += args.action_horizon
                obs = next_obs
                pbar.update(args.action_horizon)
            
            episode_subgoal_info.append(
                    (subgoal_instruction, 
                     [step, step + subtask_step]
                ))
            step += subtask_step
            
            # reset robot pose
            unwrapped = env.unwrapped
            sim = unwrapped.sim
            robot = unwrapped.robots[0]
            
            # 로봇 state 복원
            for joint, qpos_val in initial_robot_state['qpos'].items():
                joint_id = sim.model.joint_name2id(joint)
                qpos_addr = sim.model.jnt_qposadr[joint_id]
                qvel_addr = sim.model.jnt_dofadr[joint_id]
                sim.data.qpos[qpos_addr] = qpos_val
                sim.data.qvel[qvel_addr] = initial_robot_state['qvel'][joint]
            
            # 중요: MuJoCo state 동기화
            current_state = sim.get_state()
            sim.set_state(current_state)
            sim.forward()
            
            # 컨트롤러 리셋 (있는 경우에만)
            if hasattr(robot, 'controller') and robot.controller is not None:
                robot.controller.reset()
            
            # ⭐ 중요: env의 내부 상태도 업데이트
            # _obs_cache를 클리어 (None이 아니라 빈 dict로!)
            if hasattr(unwrapped, '_obs_cache'):
                unwrapped._obs_cache = {}
            
            # 새 observation 가져오기 - wrapper chain을 통과시켜야 함
            # 1. unwrapped에서 raw obs 얻기
            raw_obs = unwrapped._get_observations()
            
            # 2. RoboCasaWrapper를 찾아서 observation 변환 적용
            robocasa_wrapper = None
            multistep_wrapper = None
            current_env = env
            while hasattr(current_env, 'env'):
                if isinstance(current_env, RoboCasaWrapper):
                    robocasa_wrapper = current_env
                if isinstance(current_env, MultiStepWrapper):
                    multistep_wrapper = current_env
                current_env = current_env.env
            
            # 3. observation 변환 (RoboCasaWrapper의 로직과 동일)
            if robocasa_wrapper is not None:
                new_obs = {}
                for key, value in raw_obs.items():
                    if key in robocasa_wrapper._robocasa_keys_to_gr00t_keys:
                        if key in robocasa_wrapper._image_keys:
                            # 이미지를 numpy array로 변환 후 flip
                            value = np.asarray(value)
                            value = value[::-1]  # flip image
                        new_obs[robocasa_wrapper._robocasa_keys_to_gr00t_keys[key]] = value
                new_obs["annotation.human.action.task_description"] = [robocasa_wrapper.language_instruction]
                
                # 4. MultiStepWrapper의 internal state 업데이트
                if multistep_wrapper is not None:
                    # observation을 deque에 추가
                    multistep_wrapper.obs.append(new_obs)
                    # _get_obs()로 최종 observation 생성
                    obs = multistep_wrapper._get_obs(
                        multistep_wrapper.video_delta_indices,
                        multistep_wrapper.state_delta_indices
                    )
                else:
                    obs = new_obs
            else:
                # RoboCasaWrapper를 못 찾은 경우 raw_obs 사용 (fallback)
                obs = raw_obs
            #################################################################################    

            
        add_to(stats, flatten({"is_success": info["is_success"]}))
        subgoal_info_list.append(episode_subgoal_info)
        
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
        
        print("--------------------------------")
        print(f"Episode {i}: Success = {is_success_val} | Success rate = {success_rate:.3f}")
        
        # Check video recording status if recording
        if record_video and hasattr(env, 'recording'):
            print(f"Video recording status: {env.recording}, Frames recorded: {len(env.recorded_frames) if hasattr(env, 'recorded_frames') else 'N/A'}")
        
        pbar.close()

    # Close environment to ensure video is saved
    env.close()
    
    # Rename videos with custom format after all videos are saved
    if record_video:
        print("\n" + "="*80)
        print("Renaming video files...")
        print("="*80)
        
        renamed_count = 0
        for ep_info, subgoal_list in zip(episode_info_list, subgoal_info_list):
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
                
                # augment subgoal instruction in video
                augment_subgoal_in_video(new_video_path, subgoal_list)
                renamed_count += 1
            else:
                print(f"✗ Episode {episode_idx}: Video file not found: {old_filename}")
        
        print(f"\nRenamed {renamed_count}/{len(episode_info_list)} video files")

    # Check if videos were saved
    if record_video:
        video_files = list(video_base_path.glob("*.mp4"))
        print(f"\nVideo files saved: {len(video_files)}")
        for video_file in video_files:
            print(f"  - {video_file.name}")
        if len(video_files) == 0:
            print("⚠️  No video files were saved. Check the render method and RecordVideo wrapper.")

    for k, v in stats.items():
        stats[k] = np.mean(v)
    print(stats)

    exit()
