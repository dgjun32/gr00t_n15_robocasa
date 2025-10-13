#!/bin/bash

# Evaluation script for comparing policy predictions with ground truth
# Evaluates 8 different configurations on the last trajectory of the dataset

# Create output base directory
mkdir -p eval_results_dataset

echo "=========================================="
echo "Evaluating Policy on Dataset"
echo "Dataset: /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug"
echo "Trajectory: Last trajectory (traj_id=-1)"
echo "=========================================="
echo ""

# 1. Flow matching (original)
echo "1. Flow matching (original)..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --output_dir eval_results_dataset/1_original_flow_matching

echo ""

# 2. DDPM (diffusion, 100 steps)
echo "2. DDPM (diffusion, 100 steps)..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --diffusion_mode ddpm \
    --denoising_steps 100 \
    --output_dir eval_results_dataset/2_diffusion_ddpm_100

echo ""

# 3. DDIM (diffusion, 16 steps)
echo "3. DDIM (diffusion, 16 steps)..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --diffusion_mode ddim \
    --denoising_steps 16 \
    --output_dir eval_results_dataset/3_diffusion_ddim_16

echo ""

# 4. Regression
echo "4. Regression..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_regression_2/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --output_dir eval_results_dataset/4_regression

echo ""

# 5. Flow matching with prior_variance 0.5
echo "5. Flow matching with prior_variance 0.5..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --prior_variance 0.5 \
    --output_dir eval_results_dataset/5_original_prior_variance_0.5

echo ""

# 6. DDIM with prior_variance 0.5
echo "6. DDIM with prior_variance 0.5..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --diffusion_mode ddim \
    --denoising_steps 16 \
    --prior_variance 0.5 \
    --output_dir eval_results_dataset/6_diffusion_ddim_16_prior_variance_0.5

echo ""

# 7. Flow matching with prior_variance 0.0
echo "7. Flow matching with prior_variance 0.0..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --prior_variance 0.0 \
    --output_dir eval_results_dataset/7_original_prior_variance_0.0

echo ""

# 8. DDIM with prior_variance 0.0
echo "8. DDIM with prior_variance 0.0..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --diffusion_mode ddim \
    --denoising_steps 16 \
    --prior_variance 0.0 \
    --output_dir eval_results_dataset/8_diffusion_ddim_16_prior_variance_0.0

echo ""

# 9. DDIM with prior_variance 0.0 (steps 60-160)
echo "9. DDIM with prior_variance 0.0 (steps 60-160)..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --diffusion_mode ddim \
    --denoising_steps 16 \
    --prior_variance 0.0 \
    --prior_variance_steps 60 160 \
    --output_dir eval_results_dataset/9_diffusion_ddim_16_prior_variance_0.0_steps_60_160

echo ""

# 10. Flow matching with prior_variance 0.0 (steps 60-160)
echo "10. Flow matching with prior_variance 0.0 (steps 60-160)..."
CUDA_VISIBLE_DEVICES=2 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --dataset_path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --cfg_scale 1.0 \
    --prior_variance 0.0 \
    --prior_variance_steps 60 160 \
    --output_dir eval_results_dataset/10_original_flow_matching_prior_variance_0.0_steps_60_160

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results saved to: eval_results_dataset/"
echo "=========================================="
echo ""

# Generate summary
echo "Generating summary..."
python -c "
import json
from pathlib import Path

base_dir = Path('eval_results_dataset')
results = []

for result_dir in sorted(base_dir.iterdir()):
    if result_dir.is_dir():
        json_file = list(result_dir.glob('results_traj*.json'))
        if json_file:
            with open(json_file[0]) as f:
                data = json.load(f)
                results.append({
                    'name': result_dir.name,
                    'mse': data['overall_mse'],
                    'mse_per_dim': data.get('mse_per_dimension', {}),
                    'config': {
                        'denoising_steps': data.get('denoising_steps'),
                        'diffusion_mode': data.get('diffusion_mode'),
                        'prior_variance': data.get('prior_variance'),
                        'prior_variance_steps': data.get('prior_variance_steps'),
                        'cfg_mode': data.get('cfg_mode'),
                        'cfg_scale': data.get('cfg_scale'),
                    }
                })

if not results:
    print('No results found!')
    exit(1)

print('')
print('=' * 100)
print('SUMMARY OF RESULTS')
print('=' * 100)
print(f'{'Configuration':<60} {'Overall MSE':>15}')
print('-' * 100)
for r in sorted(results, key=lambda x: x['mse']):
    print(f'{r['name']:<60} {r['mse']:>15.6f}')
print('=' * 100)
print('')

best = min(results, key=lambda x: x['mse'])
print(f'Best configuration: {best['name']}')
print(f'Best overall MSE: {best['mse']:.6f}')
print('')

print('Per-dimension MSE for best configuration:')
for dim_name, mse_val in best['mse_per_dim'].items():
    print(f'  {dim_name}: {mse_val:.6f}')
print('')

print('Configuration details:')
for key, val in best['config'].items():
    if val is not None:
        print(f'  {key}: {val}')
print('')
"

echo "Done!"
