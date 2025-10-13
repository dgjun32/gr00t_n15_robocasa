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

"""
Evaluate policy predictions against ground truth actions from a dataset.
Computes MSE and visualizes the differences.

Based on eval_policy.py and gr00t/utils/eval.py
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy

warnings.simplefilter("ignore", category=FutureWarning)

# numpy print precision settings
np.set_printoptions(precision=3, suppress=True)


def compute_mse(pred_actions, gt_actions):
    """Compute MSE between predicted and ground truth actions."""
    diff = pred_actions - gt_actions
    squared_error = diff ** 2
    mse = squared_error.mean()
    return mse, squared_error


def plot_action_comparison(pred_actions, gt_actions, output_path, traj_id, mse, action_horizon, action_names=None):
    """Plot predicted vs ground truth actions over time."""
    num_steps, action_dim = pred_actions.shape
    
    if action_names is None:
        action_names = [f"Action {i}" for i in range(action_dim)]
    
    # Create subplots for each action dimension
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, 3 * action_dim))
    if action_dim == 1:
        axes = [axes]
    
    # Title
    fig.suptitle(f"Trajectory {traj_id} - Overall MSE: {mse:.6f}", fontsize=14, fontweight="bold")
    
    for i, ax in enumerate(axes):
        ax.plot(gt_actions[:, i], label='Ground Truth', linewidth=2, alpha=0.7)
        ax.plot(pred_actions[:, i], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
        
        # Mark inference points (every action_horizon steps)
        for j in range(0, num_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_actions[j, i], 'ro', label='Inference point', markersize=6)
            else:
                ax.plot(j, gt_actions[j, i], 'ro', markersize=4)
        
        ax.set_ylabel(action_names[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add MSE for this dimension
        mse_dim = np.mean((pred_actions[:, i] - gt_actions[:, i]) ** 2)
        ax.set_title(f"{action_names[i]} - MSE: {mse_dim:.6f}")
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_error_heatmap(squared_errors, output_path, action_names=None):
    """Plot heatmap of squared errors over time."""
    num_steps, action_dim = squared_errors.shape
    
    if action_names is None:
        action_names = [f"Action {i}" for i in range(action_dim)]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(squared_errors.T, aspect='auto', cmap='hot', interpolation='nearest')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Dimension')
    ax.set_yticks(range(action_dim))
    ax.set_yticklabels(action_names)
    ax.set_title('Squared Error Heatmap')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Squared Error')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate policy on dataset")
    
    # Model parameters
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
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="new_embodiment",
        choices=list(EMBODIMENT_TAG_MAPPING.keys()),
        help="The embodiment tag for the model.",
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug",
        help="Path to the LeRobot dataset",
    )
    parser.add_argument(
        "--traj_id",
        type=int,
        default=-1,
        help="Trajectory index to evaluate. -1 means the last trajectory.",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=16,
        help="Action horizon",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        choices=["decord", "torchvision_av"],
        default="decord",
        help="Video backend to use for various codec options. h264: decord or av: torchvision_av",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps to evaluate. If None, will use full trajectory length.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps if model_path is provided",
        default=4,
    )
    parser.add_argument(
        "--diffusion_mode",
        type=str,
        choices=["ddim", "ddpm"],
        default="ddim",
        help="Diffusion sampling mode: 'ddim' (deterministic) or 'ddpm' (stochastic).",
    )
    parser.add_argument(
        "--prior_variance",
        type=float,
        default=1.0,
        help="Variance of the prior distribution for initial actions.",
    )
    parser.add_argument(
        "--prior_variance_steps",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Step range [START, END) to apply prior_variance. Outside this range, uses standard normal (variance=1.0).",
    )
    
    # CFG parameters
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Scale factor for Conditional Free Guidance.",
    )
    parser.add_argument(
        "--cfg_mode",
        type=str,
        choices=["action", "embedding", "none"],
        default="none",
        help="CFG mode: 'action' for final action CFG, 'embedding' for model output CFG, 'none' for no CFG",
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    
    # Action selection
    parser.add_argument(
        "--modality_keys",
        nargs="+",
        type=str,
        default=['action.end_effector_position', 'action.end_effector_rotation'],
        help="Modality keys to evaluate (e.g., right_arm left_arm). If None, will infer from data config.",
    )
    parser.add_argument(
        "--exclude_gripper",
        action="store_true",
        default=True,
        help="Exclude gripper action (only consider 6D EEF)",
    )
    
    args = parser.parse_args()
    
    # Convert cfg_mode to None if "none"
    cfg_mode = None if args.cfg_mode == "none" else args.cfg_mode
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data config and transforms
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # Infer modality keys if not provided
    if args.modality_keys is None:
        args.modality_keys = list(modality_config.keys())
        # Remove non-action keys
        args.modality_keys = [k for k in args.modality_keys if 'arm' in k or 'action' in k]
        print(f"Inferred modality keys: {args.modality_keys}")
    
    # Load policy
    print("Loading policy...")
    
    # Check if we need dual policy loading for prior_variance_steps
    if args.prior_variance_steps is not None and args.prior_variance != 1.0:
        print(f"Using dual policy setup:")
        print(f"  - Base policy: prior_variance=1.0 (standard normal)")
        print(f"  - Prior policy: prior_variance={args.prior_variance}")
        print(f"  - Prior variance steps: [{args.prior_variance_steps[0]}, {args.prior_variance_steps[1]})")
        
        # Load base policy with standard normal prior
        base_policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            diffusion_mode=args.diffusion_mode,
            prior_variance=1.0,  # Standard normal
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Load prior policy with custom prior variance
        prior_policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            diffusion_mode=args.diffusion_mode,
            prior_variance=args.prior_variance,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        use_dual_policy = True
    else:
        # Single policy setup
        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            diffusion_mode=args.diffusion_mode,
            prior_variance=args.prior_variance,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        use_dual_policy = False
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    
    # Get the supported modalities for the policy
    if use_dual_policy:
        modality = base_policy.get_modality_config()
    else:
        modality = policy.get_modality_config()
    
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Total trajectories: {len(dataset.trajectory_lengths)}")
    print(f"Trajectory lengths: {dataset.trajectory_lengths}")
    
    # Select trajectory
    if args.traj_id == -1:
        traj_id = len(dataset.trajectory_lengths) - 1
    else:
        traj_id = args.traj_id
    
    print(f"\nEvaluating trajectory {traj_id}")
    traj_length = dataset.trajectory_lengths[traj_id]
    print(f"Trajectory length: {traj_length}")
    
    # Set steps
    if args.steps is None:
        steps = traj_length
    else:
        steps = min(args.steps, traj_length)
    
    print(f"Evaluating {steps} steps")
    
    # Collect predictions and ground truth
    gt_action_across_time = []
    pred_action_across_time = []
    
    print("\nRunning inference on trajectory...")
    for step_count in tqdm(range(steps)):
        data_point = dataset.get_step_data(traj_id, step_count)

        # Get ground truth action (concatenate all modality keys)
        concat_gt_action = np.concatenate(
            [data_point[f"{key}"][0] for key in args.modality_keys], axis=0
        )
        gt_action_across_time.append(concat_gt_action)
        
        # Get predicted action (only at action_horizon intervals)
        if step_count % args.action_horizon == 0:
            # Select policy based on step count for dual policy setup
            if use_dual_policy:
                start_step, end_step = args.prior_variance_steps
                if start_step <= step_count < end_step:
                    current_policy = prior_policy
                else:
                    current_policy = base_policy
            else:
                current_policy = policy
            
            if cfg_mode is None:
                action_chunk = current_policy.get_action(data_point)
            else:
                # Create empty instruction observation for CFG
                data_point_empty = {}
                for k, v in data_point.items():
                    if 'task_description' in k or 'language' in k:
                        if isinstance(v, list):
                            data_point_empty[k] = [""]
                        else:
                            data_point_empty[k] = ""
                    else:
                        data_point_empty[k] = v
                action_chunk = current_policy.get_action_cfg(data_point, data_point_empty, cfg_mode, args.cfg_scale)
            
            # Store the action chunk for the next action_horizon steps
            for j in range(args.action_horizon):
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"{key}"][j]) for key in args.modality_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)
    
    # Convert to numpy arrays
    gt_action_across_time = np.array(gt_action_across_time)
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    
    print(f"\nGround truth actions shape: {gt_action_across_time.shape}")
    print(f"Predicted actions shape: {pred_action_across_time.shape}")
    
    assert gt_action_across_time.shape == pred_action_across_time.shape, \
        f"Shape mismatch: gt {gt_action_across_time.shape} vs pred {pred_action_across_time.shape}"
    
    # Check for NaN
    if np.isnan(pred_action_across_time).any():
        print("WARNING: Predicted actions contain NaN values!")
    
    # Filter to 6D EEF (exclude gripper) if requested
    if args.exclude_gripper:
        # Assume last dimension(s) is gripper
        # For robocasa/panda: 7D action (6D EEF + 1D gripper)
        if gt_action_across_time.shape[1] >= 7:
            gt_action_across_time = gt_action_across_time[:, :6]
            pred_action_across_time = pred_action_across_time[:, :6]
            action_names = ['EEF_X', 'EEF_Y', 'EEF_Z', 'EEF_Roll', 'EEF_Pitch', 'EEF_Yaw']
            print(f"\nFiltered to 6D EEF actions (excluding gripper)")
            print(f"New shape: {gt_action_across_time.shape}")
        else:
            print(f"\nWarning: Action dimension is {gt_action_across_time.shape[1]}, cannot exclude gripper. Using all dimensions.")
            action_dim = gt_action_across_time.shape[1]
            action_names = [f"Action_{i}" for i in range(action_dim)]
    else:
        action_dim = gt_action_across_time.shape[1]
        action_names = [f"Action_{i}" for i in range(action_dim)]
    
    # Compute MSE
    mse, squared_errors = compute_mse(pred_action_across_time, gt_action_across_time)
    print(f"\n{'='*60}")
    print(f"Overall MSE: {mse:.6f}")
    print(f"{'='*60}\n")
    
    # Compute per-dimension MSE
    mse_per_dim = np.mean(squared_errors, axis=0)
    print("MSE per dimension:")
    for i, (name, mse_val) in enumerate(zip(action_names, mse_per_dim)):
        print(f"  {name}: {mse_val:.6f}")
    
    # Compute statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean: {np.mean(pred_action_across_time, axis=0)}")
    print(f"  Std:  {np.std(pred_action_across_time, axis=0)}")
    print(f"  Min:  {np.min(pred_action_across_time, axis=0)}")
    print(f"  Max:  {np.max(pred_action_across_time, axis=0)}")
    
    print(f"\nGround truth statistics:")
    print(f"  Mean: {np.mean(gt_action_across_time, axis=0)}")
    print(f"  Std:  {np.std(gt_action_across_time, axis=0)}")
    print(f"  Min:  {np.min(gt_action_across_time, axis=0)}")
    print(f"  Max:  {np.max(gt_action_across_time, axis=0)}")
    
    # Save results to JSON
    results = {
        "traj_id": int(traj_id),
        "traj_length": int(traj_length),
        "steps_evaluated": int(steps),
        "overall_mse": float(mse),
        "mse_per_dimension": {name: float(val) for name, val in zip(action_names, mse_per_dim)},
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "action_horizon": args.action_horizon,
        "denoising_steps": args.denoising_steps,
        "diffusion_mode": args.diffusion_mode,
        "prior_variance": args.prior_variance,
        "prior_variance_steps": args.prior_variance_steps,
        "cfg_mode": cfg_mode,
        "cfg_scale": args.cfg_scale if cfg_mode else None,
        "modality_keys": args.modality_keys,
    }
    
    results_path = output_dir / f"results_traj{traj_id}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Plot comparison
    print("\nGenerating plots...")
    plot_path = output_dir / f"action_comparison_traj{traj_id}.png"
    plot_action_comparison(pred_action_across_time, gt_action_across_time, plot_path, traj_id, mse, args.action_horizon, action_names)
    
    # Plot error heatmap
    heatmap_path = output_dir / f"error_heatmap_traj{traj_id}.png"
    plot_error_heatmap(squared_errors, heatmap_path, action_names)
    
    # Save raw data
    raw_data_path = output_dir / f"raw_data_traj{traj_id}.npz"
    np.savez(
        raw_data_path,
        pred_actions=pred_action_across_time,
        gt_actions=gt_action_across_time,
        squared_errors=squared_errors,
        action_names=action_names,
    )
    print(f"Saved raw data to {raw_data_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
