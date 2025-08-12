# Interact with RoboCasa Environments (Script)
#
# This script lets you interact with RoboCasa Gym environments using the same setup as `scripts/eval_policy_robocasa.py`.
#
# - Builds the environment with `RoboCasaWrapper`, `TimeLimit`, `MultiStepWrapper`, and optional `RecordVideo`.
# - Uses a local `Gr00tPolicy`.
# - Runs episodes and reports success metrics.
# - Uses argparse and saves results to a timestamped directory.

import os
import json
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from pathlib import Path

import gymnasium as gym  # noqa: F401
import robocasa  # noqa: F401
import robosuite  # noqa: F401

from gymnasium.wrappers import TimeLimit
from gr00t.eval.wrappers.robocasa_wrapper import RoboCasaWrapper, load_robocasa_gym_env
from gr00t.eval.wrappers.record_video import RecordVideo
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy, BasePolicy

def show_video_inline(mp4_path: str):  # noqa: D401
    print(f"Video saved at {mp4_path}")
    return None


@dataclass
class RunConfig:
    model_path: str
    embodiment_tag: str
    data_config_name: str
    action_horizon: int
    denoising_steps: int
    env_name: str
    seed: int
    max_episode_steps: int
    num_episodes: int
    save_video: bool
    video_folder: Optional[str]
    video_fps: int
    device: str
    layout_and_style_ids: Optional[List[Tuple[int, int]]]
    output_dir: str
    generative_textures: bool


def parse_layout_and_style_ids(value: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse a string like "1:1,2:2,4:4" into a list of (layout, style) tuples."""
    if value is None or value.strip() == "":
        return None
    pairs: List[Tuple[int, int]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid layout_and_style_ids entry '{part}'. Expected format 'layout:style'"
            )
        l, s = part.split(":", 1)
        pairs.append((int(l), int(s)))
    return pairs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interact with RoboCasa envs using a Gr00tPolicy")

    # Model/policy
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument(
        "--data_config_name",
        type=str,
        default="single_panda_gripper",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="Key into DATA_CONFIG_MAP",
    )
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--denoising_steps", type=int, default=4)

    # Environment
    parser.add_argument("--env_name", type=str, default="CloseDoubleDoor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument(
        "--layout_and_style_ids",
        type=parse_layout_and_style_ids,
        default=parse_layout_and_style_ids("1:1,2:2,4:4,6:9,7:10"),
        help="Comma-separated 'layout:style' pairs (e.g., '1:1,2:2')",
    )
    parser.add_argument("--generative_textures", action="store_true", help="Use generative textures")

    # Video
    parser.add_argument("--save_video", action="store_true", help="Enable video recording")
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="Where to save videos. Defaults to <run_dir>/videos if not set.",
    )
    parser.add_argument("--video_fps", type=int, default=20)

    # System/runtime
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.cwd() / "robocasa_runs"),
        help="Parent directory for results (a timestamped run dir is created inside)",
    )
    return parser


def select_device(arg_device: str) -> str:
    if arg_device != "auto":
        return arg_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Prepare run directory
    if args.model_path[-1] == "/":
        model_path_idx = -3
    else:
        model_path_idx = -2
    run_dir = Path(args.output_dir) / args.model_path.split('/')[model_path_idx] / args.env_name
    ensure_dir(run_dir)

    # Video folder default
    video_dir = Path(args.video_folder) if args.video_folder else run_dir / "videos"
    if args.save_video:
        ensure_dir(video_dir)

    # System env
    device = select_device(args.device)

    # Build policy
    data_config = DATA_CONFIG_MAP[args.data_config_name]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy: BasePolicy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    print("Policy modality config keys:", list(policy.get_modality_config().keys()))

    # Build environment
    rc_env = load_robocasa_gym_env(
        args.env_name,
        seed=args.seed,
        generative_textures="100p" if args.generative_textures else None,
        layout_and_style_ids=args.layout_and_style_ids,
        layout_ids=None,
        style_ids=None,
    )
    env = RoboCasaWrapper(rc_env)
    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)

    # Optional video recording
    if args.save_video:
        # Record only the last episode
        trigger = lambda ep_idx: ep_idx == args.num_episodes - 1
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=trigger,
            fps=args.video_fps,
            disable_logger=True,
        )

    # Multi-step wrapper to match policy action horizon
    env = MultiStepWrapper(
        env,
        video_delta_indices=np.arange(1),
        state_delta_indices=np.arange(1),
        n_action_steps=args.action_horizon,
    )

    print("Observation space keys:", list(env.observation_space.spaces.keys()))
    print("Action space keys:", list(env.action_space.spaces.keys()))

    # Evaluation loop
    from collections import defaultdict
    from tqdm import tqdm

    stats = defaultdict(list)
    video_paths: List[str] = []

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        done = False

        pbar = tqdm(total=args.max_episode_steps, desc=f"Episode {ep+1}", leave=False)

        while not done:
            action = policy.get_action(obs)
            # postprocess to ensure last dim exists when needed (matches eval_policy_robocasa.py)
            post_action = {}
            for k, v in action.items():
                post_action[k] = v[..., None] if getattr(v, "ndim", 0) == 1 else v

            next_obs, reward, terminated, truncated, info = env.step(post_action)
            done = terminated or truncated
            obs = next_obs
            pbar.update(args.action_horizon)

        pbar.close()

        is_success = bool(info.get("is_success", False))
        stats["is_success"].append(is_success)

        # If saving video (only last episode is recorded), stop and collect after final episode
        if args.save_video and ep == args.num_episodes - 1:
            inner = getattr(env, "env", None)
            if hasattr(inner, "stop_recording") and getattr(inner, "recording", False):
                inner.stop_recording()
            if video_dir.exists():
                files = sorted([str(p) for p in video_dir.glob("*.mp4")])
                if files:
                    video_paths.append(files[-1])

    success_rate = float(np.mean(stats["is_success"]) if stats["is_success"] else 0.0)
    print({"success_rate": success_rate})

    # Persist results
    run_config = RunConfig(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        data_config_name=args.data_config_name,
        action_horizon=args.action_horizon,
        denoising_steps=args.denoising_steps,
        env_name=args.env_name,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        num_episodes=args.num_episodes,
        save_video=bool(args.save_video),
        video_folder=str(video_dir) if args.save_video else None,
        video_fps=args.video_fps,
        device=device,
        layout_and_style_ids=args.layout_and_style_ids,
        output_dir=str(Path(args.output_dir).resolve()),
        generative_textures=args.generative_textures,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "success_rate": success_rate,
        "episode_success": stats["is_success"],
        "num_episodes": args.num_episodes,
        "video_paths": video_paths,
    }

    write_json(run_dir / "metrics.json", results)
    write_json(run_dir / "config.json", asdict(run_config))

    print(f"Saved metrics to: {run_dir / 'metrics.json'}")
    print(f"Saved config to: {run_dir / 'config.json'}")
    if args.save_video:
        print(f"Saved videos under: {video_dir}")


if __name__ == "__main__":
    main()
