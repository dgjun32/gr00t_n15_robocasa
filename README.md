# Isaac-GR00T-robocasa

This repository provides two main implementations:

- **Robocasa Simulation Environment**: Setup and run GR00T policies on Robocasa simulation environments
- **Multi-Objective Training and Inference with GR00T_N1_5**: Train and deploy policies with various action heads (Flow Matching, Diffusion, Regression)

---

<details>
<summary><b>📦 Robocasa Installation Guide</b></summary>

For setting up the robocasa simulation environment with GR00T:

### 1. GR00T Environment Setup

Clone the robocasa-compatible repository:

```sh
git clone https://github.com/sinnnj/Issac-GR00T-robocasa.git 
cd Isaac-GR00T-robocasa
conda create -n gr00t_rc python=3.10
conda activate gr00t_rc
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4 
```

### 2. Robosuite and Robocasa Installation

```sh
git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .
cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa
pip install -e .
```

### 3. Download Assets

```sh
# Caution: Assets to be downloaded are around 5GB
python robocasa/scripts/download_kitchen_assets.py   
python robocasa/scripts/setup_macros.py              # Set up system variables
```

### 4. Troubleshooting

#### For VAST Platform:

```sh
pip install protobuf==3.20.3
pip install tianshou==0.5.1
pip install gymnasium[other]
pip install numpy==1.23.3

# For OSMesa rendering
apt-get install -y libosmesa6-dev
export MUJOCO_GL=osmesa

# For EGL rendering
apt-get update && apt-get install -y libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev
```

#### For Alin slurm Platform:

```sh
conda install --force-reinstall pytz
conda install --force-reinstall six
conda install --force-reinstall idna
conda install --force-reinstall certifi
pip install --upgrade protobuf
```
</details>

<details>
<summary><b>🚀 Example Robocasa Evaluation Command</b></summary>

After installation, you can run robocasa evaluation with the following example command:

```sh
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa_lh.py \
    --model_path checkpoints/checkpoint-40000 \
    --action_horizon 8 \
    --planner_model_name gemini-robotics-er-1.5-preview \
    --subgoal_interval 200 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name OpenDrawer \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_test/OpenDrawer \
    --max_episode_steps 1000


CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_policy_robocasa.py \
    --model_path checkpoints/checkpoint-60000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name PnPCabToCounter \
    --num_episodes 10 \
    --video_path /home/dongjun/Isaac-GR00T-robocasa/eval_test/PnPCabToCounter \
    --max_episode_steps 1000
```

</details>

---

<details>
<summary><b>🎯 Multi-Objective Training and Inference with GR00T_N1_5</b></summary>

This repository provides a flexible framework that enables training and inference with various training objectives while maintaining the same architecture. Simply by changing the action head type, you can train and deploy policies using different approaches such as Flow Matching, Diffusion, or Regression.

### 1. How to Switch Action Heads

The action head implementations can be found in the following files:
- `gr00t/model/action_head/flow_matching_action_head.py` - Flow Matching approach (default)
- `gr00t/model/action_head/diffusion_action_head.py` - DDPM/DDIM Diffusion approach
- `gr00t/model/action_head/regression_action_head.py` - Direct Regression approach

#### Config-based Automatic Action Head Creation

Simply set `action_head_type` in `GR00T_N1_5_Config` and the appropriate action head will be automatically created:

```python
# gr00t/model/gr00t_n1.py

@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    action_head_type: str = field(
        default="flowmatching", 
        metadata={"help": "Action head type: 'flowmatching', 'diffusion', or 'regression'."}
    )
    # ... other configs

def create_action_head_from_config(action_head_cfg_dict: dict):
    """Dynamically create the appropriate action head based on config"""
    action_head_type = action_head_cfg_dict.get("action_head_type", None)
    
    if action_head_type == "regression":
        action_head_cfg = RegressionActionHeadConfig(**action_head_cfg_dict)
        return RegressionActionHead(action_head_cfg)
    elif action_head_type == "diffusion":
        action_head_cfg = DiffusionActionHeadConfig(**action_head_cfg_dict)
        return DiffusionActionHead(action_head_cfg)
    else:
        action_head_cfg = FlowmatchingActionHeadConfig(**action_head_cfg_dict)
        return FlowmatchingActionHead(action_head_cfg)
```

#### Automatic Application During Fine-tuning

During training with `scripts/gr00t_finetune.py`, simply setting the `--action_head_type` parameter will automatically configure the desired action head through `GR00T_N1_5.from_pretrained()`:

```bash
# Example: Train with diffusion action head
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/dataset \
    --action_head_type diffusion \
    ...
```

The `action_head_type` parameter is passed to `GR00T_N1_5.from_pretrained()`, which automatically loads the appropriate action head:

```python
# gr00t/model/gr00t_n1.py

@classmethod
def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
    action_head_type = kwargs.pop("action_head_type", None)
    
    config = cls.config_class.from_pretrained(local_model_path)
    if hasattr(config, 'action_head_cfg') and isinstance(config.action_head_cfg, dict):
        if action_head_type is not None:
            config.action_head_cfg['action_head_type'] = action_head_type
    
    pretrained_model = super().from_pretrained(
        local_model_path, local_model_path=local_model_path, config=config, **kwargs
    )
    
    print("Loaded action head type : ", type(pretrained_model.action_head).__name__)
    return pretrained_model
```

#### Automatic Config Loading During Inference

The `_load_model()` function in `Gr00tPolicy` reads the saved config from the checkpoint and automatically applies the appropriate action head:

```python
# gr00t/model/policy.py

def _load_model(self, model_path):
    model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
    
    # Read action head information from saved config
    new_action_head_cfg_dict = dict(model.config.action_head_cfg)
    
    # create_action_head_from_config automatically determines and creates the type
    new_action_head = create_action_head_from_config(new_action_head_cfg_dict)
    
    # Load existing weights into the new action head
    new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
    model.action_head = new_action_head
    
    print("Updated action head type: ", type(model.action_head).__name__)
    self.model = model
```

---

### Implementaion Detail : Diffusion Action Head

The diffusion action head generates actions based on DDPM (Denoising Diffusion Probabilistic Model).

#### Training Objective

During training, the model learns to recover actions by adding noise to clean actions:

```python
# gr00t/model/action_head/diffusion_action_head.py - forward()

# Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
actions = action_input.action
noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
t_discretized = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)

sqrt_alpha_prod = self.sqrt_alphas_cumprod[t_discretized][:, None, None]
sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t_discretized][:, None, None]

noisy_trajectory = sqrt_alpha_prod * actions + sqrt_one_minus_alpha_prod * noise

# Target: predict noise (epsilon prediction)
if self.prediction_type == "epsilon":
    target = noise
elif self.prediction_type == "sample":
    target = actions

# Encode noisy action with timestep information
action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

# ... model forward pass ...

pred_actions = self.action_decoder(model_output, embodiment_id)
loss = F.mse_loss(pred_actions, target, reduction="none") * action_mask
```

#### Inference (get_action)

During inference, the model performs iterative denoising starting from random noise:


```python
# gr00t/model/action_head/diffusion_action_head.py - get_action()

# 1. Initialize with random noise
actions = torch.randn(
    size=(batch_size, self.config.action_horizon, self.config.action_dim),
    dtype=vl_embs.dtype, device=device
) * self.prior_std

# 2. Create timestep schedule for reverse process
timestep_indices = torch.linspace(
    self.num_timestep_buckets - 1, 0, num_steps, dtype=torch.long, device=device
)

# 3. Iterative denoising loop
for i, t_idx in enumerate(timestep_indices):
    timesteps_tensor = torch.full(
        size=(batch_size,), fill_value=t_idx, device=device, dtype=torch.long
    )
    
    # Encode current noisy action with timestep
    action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
    
    # Predict noise or x0
    model_output = self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=timesteps_tensor)
    pred_noise_or_x0 = self.action_decoder(model_output, embodiment_id)
    
    # Recover clean action and update for next step
    if self.prediction_type == "epsilon":
        pred_x0 = (actions - sqrt_one_minus_alpha_prod_t * pred_noise_or_x0) / sqrt_alpha_prod_t
    
    # DDPM: stochastic sampling with noise
    # DDIM: deterministic sampling without noise
    if self.use_ddim:
        actions = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * epsilon_t
    else:
        actions = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * noise

return actions
```

- **Choice between DDPM (stochastic) or DDIM (deterministic) sampling**: The diffusion action head supports both DDPM for stochastic sampling and DDIM for deterministic sampling.
- **Support for cosine or linear noise schedules**: You can configure the noise schedule type in the action head config.

---

### Implementaion Detail : Regression Action Head

The regression action head directly predicts actions.

#### Training Objective

During training, the model learns to directly predict ground truth actions:

```python
# gr00t/model/action_head/regression_action_head.py - forward()

# Ground truth actions
gt_actions = action_input.action

# Dummy action initialization (ones, not ground truth!)
dummy_actions = torch.ones_like(gt_actions)
t_discretized = torch.zeros(gt_actions.shape[0], device=device, dtype=torch.long)

# Encode dummy action (timestep always 0 for regression)
action_features = self.action_encoder(dummy_actions, t_discretized, embodiment_id)

# ... model forward pass ...

pred = self.action_decoder(model_output, embodiment_id)
pred_actions = pred[:, -gt_actions.shape[1] :]

# Direct MSE loss with ground truth
loss = F.mse_loss(pred_actions, gt_actions, reduction="none") * action_mask
```

#### Inference (get_action)

During inference, actions are directly generated with a single forward pass:

```python
# gr00t/model/action_head/regression_action_head.py - get_action()

# 1. Initialize with ones (non-zero activation)
actions = torch.ones(
    size=(batch_size, self.config.action_horizon, self.config.action_dim),
    dtype=vl_embs.dtype, device=device
)

# 2. Single forward pass (timestep always 0)
timesteps_tensor = torch.zeros(batch_size, device=device, dtype=torch.long)
action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

# 3. Direct action prediction
model_output = self.model(
    hidden_states=sa_embs, 
    encoder_hidden_states=vl_embs, 
    timestep=timesteps_tensor
)
pred_actions = self.action_decoder(model_output, embodiment_id)[:, -self.action_horizon :]

return pred_actions
```

---
</details>

<details>
<summary><b>🔧 Multi-objective Training Commands</b></summary>

#### Diffusion Action Head Training

```sh
CUDA_VISIBLE_DEVICES=1 python scripts/gr00t_finetune.py \
    --dataset-path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --num-gpus 1 \
    --output-dir /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion \
    --max-steps 20000 \
    --data-config single_panda_gripper \
    --batch-size 32 \
    --save-steps 1000 \
    --action_head_type diffusion \
    --learning-rate 1e-4 \
    --weight-decay 1e-6 \
    --warmup-ratio 0.02 \
    2>&1 | tee ./training_logs/training_$(date +%Y%m%d_%H%M%S).log
```

#### Regression Action Head Training

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
    --dataset-path /data/home_backup_sj/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --num-gpus 1 \
    --output-dir /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_regression \
    --max-steps 20000 \
    --data-config single_panda_gripper \
    --batch-size 32 \
    --save-steps 1000 \
    --action_head_type regression \
    2>&1 | tee ./training_logs/training_$(date +%Y%m%d_%H%M%S).log
```

</details>


<details>
<summary><b>🚀 Multi-objective Inference Commands</b></summary>

After training with different action heads, you can run inference using various denoising configurations.

**Note**: You can also evaluate your trained models on a validation dataset using `scripts/eval_policy_on_dataset.py`. This allows you to compare predicted actions against ground truth and compute metrics such as MSE. See `scripts/run_eval_on_dataset.sh` for batch evaluation examples.

#### Flow Matching Inference (Default)

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_robocasa_cfg.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeeServeMug \
    --num_episodes 50 \
    --video_path /home/sinjae/Isaac-GR00T-robocasa/eval_video/CoffeeServeMug_flow_matching \
    --max_episode_steps 1000 \
    --cfg_mode none \
    2>&1 | tee ./logs/flow_matching_$(date +%Y%m%d_%H%M%S).log
```

#### Diffusion Inference with DDPM

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_robocasa_cfg.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeeServeMug \
    --num_episodes 50 \
    --video_path /home/sinjae/Isaac-GR00T-robocasa/eval_video/CoffeeServeMug_ddpm \
    --max_episode_steps 1000 \
    --cfg_mode none \
    --diffusion_mode ddpm \
    --denoising_steps 100 \
    2>&1 | tee ./logs/diffusion_ddpm_$(date +%Y%m%d_%H%M%S).log
```

#### Diffusion Inference with DDIM

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_robocasa_cfg.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeeServeMug \
    --num_episodes 50 \
    --video_path /home/sinjae/Isaac-GR00T-robocasa/eval_video/CoffeeServeMug_ddim \
    --max_episode_steps 1000 \
    --cfg_mode none \
    --diffusion_mode ddim \
    --denoising_steps 16 \
    2>&1 | tee ./logs/diffusion_ddim_$(date +%Y%m%d_%H%M%S).log
```

#### Regression Inference

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_robocasa_cfg.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_regression/checkpoint-20000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeeServeMug \
    --num_episodes 50 \
    --video_path /home/sinjae/Isaac-GR00T-robocasa/eval_video/CoffeeServeMug_regression \
    --max_episode_steps 1000 \
    --cfg_mode none \
    2>&1 | tee ./logs/regression_$(date +%Y%m%d_%H%M%S).log
```

#### Advanced: Inference with Prior Variance Control

You can control the prior variance during inference to adjust the exploration-exploitation trade-off:

```sh
# Flow matching with prior variance
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_robocasa_cfg.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeeServeMug \
    --num_episodes 50 \
    --video_path /home/sinjae/Isaac-GR00T-robocasa/eval_video/CoffeeServeMug_prior_variance_0.5 \
    --max_episode_steps 1000 \
    --cfg_mode none \
    --prior_variance 0.5 \
    2>&1 | tee ./logs/prior_variance_0.5_$(date +%Y%m%d_%H%M%S).log

# Flow matching with step-specific prior variance
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_robocasa_cfg.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --action_horizon 16 \
    --video_backend decord \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --env_name CoffeeServeMug \
    --num_episodes 50 \
    --video_path /home/sinjae/Isaac-GR00T-robocasa/eval_video/CoffeeServeMug_steps_60_160 \
    --max_episode_steps 1000 \
    --cfg_mode none \
    --prior_variance 0.0 \
    --prior_variance_steps 60 160 \
    2>&1 | tee ./logs/steps_60_160_$(date +%Y%m%d_%H%M%S).log
```

#### Offline Evaluation on Validation Dataset

You can evaluate your trained models on a validation dataset to compare predicted actions with ground truth:

```sh
# Flow Matching evaluation on dataset
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_original/checkpoint-20000 \
    --dataset_path /data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --output_dir eval_results_dataset/flow_matching

# Diffusion DDIM evaluation on dataset
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_diffusion/checkpoint-20000 \
    --dataset_path /data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --diffusion_mode ddim \
    --denoising_steps 16 \
    --output_dir eval_results_dataset/diffusion_ddim_16

# Regression evaluation on dataset
CUDA_VISIBLE_DEVICES=0 python scripts/eval_policy_on_dataset.py \
    --model_path /home/sinjae/Isaac-GR00T-robocasa/gr00t_ckpt/1011/CoffeeServeMug_regression/checkpoint-20000 \
    --dataset_path /data/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/single_panda_gripper.CoffeeServeMug \
    --traj_id -1 \
    --embodiment_tag new_embodiment \
    --data_config single_panda_gripper \
    --action_horizon 16 \
    --video_backend decord \
    --cfg_mode none \
    --output_dir eval_results_dataset/regression
```

The evaluation results will be saved in the specified output directory, including action comparison plots and MSE metrics.

</details>
</details>
