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
import os
import warnings
import tempfile
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robocasa
import robosuite
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium.wrappers import TimeLimit
from robosuite.controllers import load_composite_controller_config
from tqdm import tqdm, trange

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy

warnings.simplefilter("ignore", category=FutureWarning)


class EmptyTextRoboCasaWrapper(RoboCasaWrapper):
    """RoboCasa wrapper that returns empty text instruction."""
    
    @property
    def language_instruction(self):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="single_panda_gripper",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--video_backend", type=str, default="decord")
    parser.add_argument("--dataset_path", type=str, default="0801_robocasa_multitask_sj_300")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps",
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the robocasa environment",
    )
    parser.add_argument(
        "--max_steps_per_task",
        type=int,
        default=10,
        help="Maximum number of steps to run per task for embedding analysis",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embedding_analysis",
        help="Directory to save analysis results",
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

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for data collection
    temp_dir = Path(tempfile.mkdtemp(prefix="eagle_embedding_analysis_"))

    # 8 RoboCasa tasks to analyze
    TASKS = [
        "CloseDoubleDoor",
        "OpenDoubleDoor", 
        "CoffeeServeMug",
        "CoffeeSetupMug",
        "PnPCabToCounter",
        "PnPCounterToCab",
        "TurnOffSinkFaucet",
        "TurnOnSinkFaucet"
    ]

    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Initialize policy with embedding saving enabled
    policy: BasePolicy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Enable embedding saving in the backbone
    if hasattr(policy.model, 'backbone') and hasattr(policy.model.backbone, 'save_embeddings'):
        policy.model.backbone.save_embeddings = True
    else:
        print("Warning: Model backbone doesn't support embedding saving")
        return

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    all_embeddings = {}
    all_empty_text_embeddings = {}

    print(f"Analyzing eagle embeddings for {len(TASKS)} tasks...")
    
    for task_idx, task_name in enumerate(TASKS):
        print(f"\n=== Processing Task {task_idx + 1}/{len(TASKS)}: {task_name} ===")
        
        # Clear previous embeddings
        policy.model.backbone.clear_saved_embeddings()
        
        ##### Create environment for this task #####
        env = load_robocasa_gym_env(
            task_name,
            seed=args.seed,
            directory=temp_dir / f"{task_name}_regular",
            generative_textures=None,
        )
        env = RoboCasaWrapper(env)
        env = TimeLimit(env, max_episode_steps=1000)
        env = MultiStepWrapper(
            env,
            video_delta_indices=np.arange(1),
            state_delta_indices=np.arange(1),
            n_action_steps=args.action_horizon,
        )

        # Run for max_steps_per_task steps to collect embeddings
        obs, info = env.reset()
        task_embeddings = []
        
        for step in range(args.max_steps_per_task):
            print(f"Task {task_name}, Step {step + 1}/{args.max_steps_per_task}")
            
            # Get action and extract embeddings during forward pass
            action = policy.get_action(obs)
            
            # Get the last saved embedding (from this forward pass)
            if policy.model.backbone.saved_embeddings:
                current_embedding = policy.model.backbone.saved_embeddings[-1]
                task_embeddings.append(current_embedding)
            
            # Step the environment
            post_action = {}
            for k, v in action.items():
                if v.ndim == 1:
                    post_action[k] = v[..., None]
                else:
                    post_action[k] = v
            
            next_obs, reward, terminated, truncated, info = env.step(post_action)
            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs
        
        env.close()
        
        # Store embeddings for this task
        if task_embeddings:
            all_embeddings[task_name] = torch.cat(task_embeddings, dim=0)
            print(f"Collected {len(task_embeddings)} embeddings for {task_name}")
        
        ##### Now collect embeddings with empty text instruction #####
        print(f"Collecting embeddings with empty text for {task_name}...")
        policy.model.backbone.clear_saved_embeddings()
        
        # Create environment again with empty text
        env = load_robocasa_gym_env(
            task_name,
            seed=args.seed,
            directory=temp_dir / f"{task_name}_empty_text",
            generative_textures=None,
        )
        env = EmptyTextRoboCasaWrapper(env)  # Use empty text wrapper
        env = TimeLimit(env, max_episode_steps=1000)
        env = MultiStepWrapper(
            env,
            video_delta_indices=np.arange(1),
            state_delta_indices=np.arange(1),
            n_action_steps=args.action_horizon,
        )

        obs, info = env.reset()
        empty_text_embeddings = []
        
        for step in range(args.max_steps_per_task):
            print(f"Empty text {task_name}, Step {step + 1}/{args.max_steps_per_task}")
            
            # Get action and extract embeddings
            action = policy.get_action(obs)
            
            if policy.model.backbone.saved_embeddings:
                current_embedding = policy.model.backbone.saved_embeddings[-1]
                empty_text_embeddings.append(current_embedding)
            
            # Step the environment
            post_action = {}
            for k, v in action.items():
                if v.ndim == 1:
                    post_action[k] = v[..., None]
                else:
                    post_action[k] = v
            
            next_obs, reward, terminated, truncated, info = env.step(post_action)
            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs
        
        env.close()
        
        if empty_text_embeddings:
            all_empty_text_embeddings[task_name] = torch.cat(empty_text_embeddings, dim=0)
            print(f"Collected {len(empty_text_embeddings)} empty-text embeddings for {task_name}")

    # Analysis and visualization
    print("\n=== Analyzing Embeddings ===")
    
    # 1. Compute cosine similarity matrix between all task embeddings
    task_mean_embeddings = {}
    empty_text_mean_embeddings = {}
    
    for task_name in TASKS:
        if task_name in all_embeddings:
            task_mean_embeddings[task_name] = all_embeddings[task_name].mean(dim=0)
        if task_name in all_empty_text_embeddings:
            empty_text_mean_embeddings[task_name] = all_empty_text_embeddings[task_name].mean(dim=0)
    
    # Create similarity matrix between tasks
    task_names = list(task_mean_embeddings.keys())
    n_tasks = len(task_names)
    
    if n_tasks > 0:
        # Inter-task similarity matrix
        task_embeddings_tensor = torch.stack([task_mean_embeddings[name] for name in task_names])
        task_embeddings_normalized = F.normalize(task_embeddings_tensor, p=2, dim=1)
        inter_task_similarity = torch.mm(task_embeddings_normalized, task_embeddings_normalized.t())
        
        # Plot inter-task similarity matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            inter_task_similarity.numpy(),
            xticklabels=task_names,
            yticklabels=task_names,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            vmin=-1, vmax=1
        )
        plt.title('Inter-Task Cosine Similarity Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'inter_task_similarity.png', dpi=300)
        plt.close()
        
        # Save similarity matrix as numpy array
        np.save(output_dir / 'inter_task_similarity.npy', inter_task_similarity.numpy())
        print(f"Inter-task similarity matrix saved to {output_dir / 'inter_task_similarity.npy'}")
    
    # 2. Compare task embeddings with empty text embeddings
    if task_mean_embeddings and empty_text_mean_embeddings:
        empty_text_names = list(empty_text_mean_embeddings.keys())
        n_empty = len(empty_text_names)
        
        if n_empty > 0:
            # Task vs empty text similarity
            task_vs_empty_similarity = torch.zeros(n_tasks, n_empty)
            
            for i, task_name in enumerate(task_names):
                for j, empty_name in enumerate(empty_text_names):
                    if task_name in task_mean_embeddings and empty_name in empty_text_mean_embeddings:
                        task_emb = F.normalize(task_mean_embeddings[task_name].unsqueeze(0), p=2, dim=1)
                        empty_emb = F.normalize(empty_text_mean_embeddings[empty_name].unsqueeze(0), p=2, dim=1)
                        similarity = torch.mm(task_emb, empty_emb.t()).item()
                        task_vs_empty_similarity[i, j] = similarity
            
            # Plot task vs empty text similarity
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                task_vs_empty_similarity.numpy(),
                xticklabels=empty_text_names,
                yticklabels=task_names,
                annot=True,
                fmt='.3f',
                cmap='coolwarm',
                vmin=-1, vmax=1
            )
            plt.title('Task Embeddings vs Empty Text Embeddings Cosine Similarity')
            plt.xlabel('Empty Text Tasks')
            plt.ylabel('Regular Tasks')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / 'task_vs_empty_text_similarity.png', dpi=300)
            plt.close()
            
            # Save similarity matrix
            np.save(output_dir / 'task_vs_empty_text_similarity.npy', task_vs_empty_similarity.numpy())
            print(f"Task vs empty text similarity matrix saved to {output_dir / 'task_vs_empty_text_similarity.npy'}")
            
            # Empty text vs empty text similarity
            empty_text_embeddings_tensor = torch.stack([empty_text_mean_embeddings[name] for name in empty_text_names])
            empty_text_embeddings_normalized = F.normalize(empty_text_embeddings_tensor, p=2, dim=1)
            empty_text_similarity = torch.mm(empty_text_embeddings_normalized, empty_text_embeddings_normalized.t())
            
            # Plot empty text vs empty text similarity
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                empty_text_similarity.numpy(),
                xticklabels=empty_text_names,
                yticklabels=empty_text_names,
                annot=True,
                fmt='.3f',
                cmap='coolwarm',
                vmin=-1, vmax=1
            )
            plt.title('Empty Text Tasks Inter-Similarity Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / 'empty_text_similarity.png', dpi=300)
            plt.close()
            
            # Save empty text similarity matrix
            np.save(output_dir / 'empty_text_similarity.npy', empty_text_similarity.numpy())
            print(f"Empty text similarity matrix saved to {output_dir / 'empty_text_similarity.npy'}")
    
    # 3. Intra-task similarity analysis (similarity within each task's steps)
    intra_task_similarities = {}
    
    for task_name in TASKS:
        if task_name in all_embeddings and all_embeddings[task_name].shape[0] > 1:
            embeddings = all_embeddings[task_name]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
            
            # Get upper triangular part (excluding diagonal)
            triu_indices = torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], offset=1)
            upper_similarities = similarity_matrix[triu_indices[0], triu_indices[1]]
            
            intra_task_similarities[task_name] = {
                'mean': upper_similarities.mean().item(),
                'std': upper_similarities.std().item(),
                'min': upper_similarities.min().item(),
                'max': upper_similarities.max().item()
            }
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nInter-task similarity statistics:")
    if n_tasks > 0:
        # Get upper triangular part of inter-task similarity (excluding diagonal)
        triu_indices = torch.triu_indices(inter_task_similarity.shape[0], inter_task_similarity.shape[1], offset=1)
        inter_task_upper = inter_task_similarity[triu_indices[0], triu_indices[1]]
        
        print(f"  Mean: {inter_task_upper.mean().item():.4f}")
        print(f"  Std:  {inter_task_upper.std().item():.4f}")
        print(f"  Min:  {inter_task_upper.min().item():.4f}")
        print(f"  Max:  {inter_task_upper.max().item():.4f}")
    
    print("\nIntra-task similarity statistics:")
    for task_name, stats in intra_task_similarities.items():
        print(f"  {task_name}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std:  {stats['std']:.4f}")
        print(f"    Min:  {stats['min']:.4f}")
        print(f"    Max:  {stats['max']:.4f}")
    
    # Save summary statistics
    summary_stats = {
        'inter_task_similarity': {
            'mean': inter_task_upper.mean().item() if n_tasks > 0 else None,
            'std': inter_task_upper.std().item() if n_tasks > 0 else None,
            'min': inter_task_upper.min().item() if n_tasks > 0 else None,
            'max': inter_task_upper.max().item() if n_tasks > 0 else None,
        },
        'intra_task_similarities': intra_task_similarities
    }
    
    import json
    with open(output_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Clean up temporary directory
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"Files generated:")
    print(f"  - inter_task_similarity.png")
    print(f"  - inter_task_similarity.npy") 
    print(f"  - task_vs_empty_text_similarity.png")
    print(f"  - task_vs_empty_text_similarity.npy")
    print(f"  - empty_text_similarity.png")
    print(f"  - empty_text_similarity.npy")
    print(f"  - summary_statistics.json")


if __name__ == "__main__":
    main() 