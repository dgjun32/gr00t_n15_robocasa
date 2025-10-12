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

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class DiffusionActionHeadConfig(PretrainedConfig):
    """Diffusion-based action head configuration"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    
    # Diffusion-specific parameters
    noise_schedule_type: str = field(
        default="cosine", metadata={"help": "Noise schedule type: 'linear' or 'cosine'"}
    )
    beta_start: float = field(default=0.0001, metadata={"help": "Starting beta for linear schedule"})
    beta_end: float = field(default=0.02, metadata={"help": "Ending beta for linear schedule"})
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=200,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    prediction_type: str = field(
        default="epsilon", metadata={"help": "Prediction type: 'epsilon' (noise) or 'sample' (x0)"}
    )
    
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class DiffusionActionHead(nn.Module):
    config_class = DiffusionActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: DiffusionActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # Diffusion noise schedule
        self.num_timestep_buckets = config.num_timestep_buckets
        self.noise_schedule_type = config.noise_schedule_type
        self.prediction_type = config.prediction_type
        
        # Precompute noise schedule
        self._initialize_noise_schedule(config)
        
        # Sampling method: DDIM (deterministic) or DDPM (stochastic)
        self.use_ddim = False  # Set to False for DDPM stochastic sampling
        
        # Prior distribution std (default: 1.0 for standard normal)
        # Can be overridden at runtime via policy.py
        self.prior_std = 1.0
        
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)
        
        # Model output saving functionality
        self.save_model_outputs = False
        self.saved_model_outputs = []
        
        # Velocity tracking functionality for CFG analysis
        self.save_velocity_analysis = True
        self.velocity_analysis_data = {
            'dt_values': [],
            'pred_noises': [],
            'pred_noises_uncond': [],
            'noise_cos_sims': [],
            'cond_uncond_cos_sims': []
        }

    def _initialize_noise_schedule(self, config):
        """Initialize diffusion noise schedule (alpha_bar_t)"""
        timesteps = config.num_timestep_buckets
        
        if config.noise_schedule_type == "linear":
            # Linear beta schedule
            betas = torch.linspace(config.beta_start, config.beta_end, timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
        elif config.noise_schedule_type == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021)
            def cosine_schedule(t):
                s = 0.008  # offset
                f_t = torch.cos(((t / timesteps + s) / (1 + s)) * torch.pi / 2) ** 2
                return f_t
            
            steps = torch.arange(timesteps + 1, dtype=torch.float32)
            alphas_cumprod = cosine_schedule(steps) / cosine_schedule(torch.tensor([0.0]))
            alphas_cumprod = alphas_cumprod[1:]  # Remove the extra element
            
        else:
            raise ValueError(f"Unknown noise schedule type: {config.noise_schedule_type}")
        
        # Register as buffers (will be moved to device with model)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Sample random timesteps uniformly"""
        return torch.randint(0, self.num_timestep_buckets, (batch_size,), device=device, dtype=torch.long)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # === DIFFUSION: Add noise to clean actions ===
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        
        # Sample random timesteps
        t_discretized = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        
        # Get noise schedule parameters
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t_discretized][:, None, None]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t_discretized][:, None, None]
        
        # Forward diffusion process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy_trajectory = sqrt_alpha_prod * actions + sqrt_one_minus_alpha_prod * noise
        
        # Target depends on prediction type
        if self.prediction_type == "epsilon":
            target = noise  # Predict noise
        elif self.prediction_type == "sample":
            target = actions  # Predict clean action (x0)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Embed noisy action trajectory
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        
        # Save model output if requested (mean pooled for easier comparison)
        if self.save_model_outputs:
            if model_output.dim() == 3:  # [B, seq_len, hidden_dim]
                pooled_model_output = model_output.mean(dim=1)  # [B, hidden_dim]
                self.saved_model_outputs.append(pooled_model_output.detach().cpu())
            else:
                self.saved_model_outputs.append(model_output.detach().cpu())
        
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, target, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        ) * self.prior_std  # Apply prior std

        num_steps = self.num_inference_timesteps
        # num_steps = 10 # for testing
        
        # DDPM/DDIM sampling
        # Create timestep schedule for inference
        timestep_indices = torch.linspace(self.num_timestep_buckets - 1, 0, num_steps, dtype=torch.long, device=device)

        # Run denoising steps (reverse process)
        print(f"Running {num_steps} denoising steps...", flush=True)
        for i, t_idx in enumerate(timestep_indices):
            t_idx = t_idx.item()
            
            # Current timestep tensor
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_idx, device=device, dtype=torch.long
            )
            
            # Embed noised action trajectory
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            
            # Maybe add position embedding
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            
            # Save model output if requested
            if self.save_model_outputs:
                if model_output.dim() == 3:  # [B, seq_len, hidden_dim]
                    pooled_model_output = model_output.mean(dim=1)  # [B, hidden_dim]
                    self.saved_model_outputs.append(pooled_model_output.detach().cpu())
                else:
                    self.saved_model_outputs.append(model_output.detach().cpu())

            pred = self.action_decoder(model_output, embodiment_id)
            pred_noise_or_x0 = pred[:, -self.action_horizon :]

            # DDPM update rule
            alpha_prod_t = self.alphas_cumprod[t_idx]
            
            # Add small epsilon to prevent division by zero
            eps = 1e-8
            
            if self.prediction_type == "epsilon":
                # Clamp predicted noise to prevent extreme values
                pred_noise_or_x0 = torch.clamp(pred_noise_or_x0, -10.0, 10.0)
                
                # Predict x0 from noise prediction (with epsilon for numerical stability)
                sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t.clamp(min=eps))
                sqrt_one_minus_alpha_prod_t = torch.sqrt((1 - alpha_prod_t).clamp(min=eps))
                
                pred_x0 = (actions - sqrt_one_minus_alpha_prod_t * pred_noise_or_x0) / sqrt_alpha_prod_t
            elif self.prediction_type == "sample":
                # Direct x0 prediction
                pred_x0 = pred_noise_or_x0
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
            # Clamp predicted x0 to action space (critical for stability!)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Get next timestep
            if i < len(timestep_indices) - 1:
                next_t_idx = timestep_indices[i + 1].item()
                alpha_prod_t_prev = self.alphas_cumprod[next_t_idx]
                
                sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t.clamp(min=eps))
                sqrt_one_minus_alpha_prod_t = torch.sqrt((1 - alpha_prod_t).clamp(min=eps))
                sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev.clamp(min=eps))
                sqrt_one_minus_alpha_prod_t_prev = torch.sqrt((1 - alpha_prod_t_prev).clamp(min=eps))
                
                if self.use_ddim:
                    # DDIM (deterministic): use predicted direction
                    epsilon_t = (actions - sqrt_alpha_prod_t * pred_x0) / sqrt_one_minus_alpha_prod_t
                    actions = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * epsilon_t
                else:
                    # DDPM (stochastic): add random noise
                    noise = torch.randn_like(actions)
                    actions = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * noise
                
                # Clamp actions to prevent explosion
                actions = torch.clamp(actions, -10.0, 10.0)
            else:
                # Final step: return predicted x0
                actions = pred_x0
        # print(f"actions: {actions}", flush=True)
        return BatchFeature(data={"action_pred": actions})

    @torch.no_grad()
    def get_action_cfg(self, backbone_output: BatchFeature, action_input: BatchFeature, 
                      backbone_output_uncond: BatchFeature, cfg_mode: str = None, cfg_scale: float = 2.0) -> BatchFeature:
        """
        Get action with Conditional Free Guidance.
        
        Args:
            backbone_output: Conditional backbone output (with instruction)
            action_input: Action input
            backbone_output_uncond: Unconditional backbone output (empty instruction)
            cfg_mode: "action" for final action CFG, "embedding" for model output CFG, None for no CFG
            cfg_scale: CFG scale factor
            
        Returns:
            BatchFeature with action prediction
        """
        if cfg_mode is None:
            # No CFG, use regular get_action
            return self.get_action(backbone_output, action_input)
        
        # Clear previous velocity analysis data if tracking is enabled
        if self.save_velocity_analysis:
            self.clear_velocity_analysis()
        
        # Process both conditional and unconditional backbone outputs
        backbone_output = self.process_backbone_output(backbone_output)
        backbone_output_uncond = self.process_backbone_output(backbone_output_uncond)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_embs_uncond = backbone_output_uncond.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        ) * self.prior_std  # Apply prior std

        num_steps = self.num_inference_timesteps
        
        # Create timestep schedule for inference
        timestep_indices = torch.linspace(self.num_timestep_buckets - 1, 0, num_steps, dtype=torch.long, device=device)

        # Run denoising steps (reverse process)
        for i, t_idx in enumerate(timestep_indices):
            t_idx = t_idx.item()
            
            # Current timestep tensor
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_idx, device=device, dtype=torch.long
            )
            
            # Embed noised action trajectory
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            
            # Maybe add position embedding
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward for conditional
            model_output_cond = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            
            # Run model forward for unconditional
            model_output_uncond = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_uncond,
                timestep=timesteps_tensor,
            )
            
            # Save model output if requested (only save conditional output to avoid duplication)
            if self.save_model_outputs:
                if model_output_cond.dim() == 3:  # [B, seq_len, hidden_dim]
                    pooled_model_output = model_output_cond.mean(dim=1)  # [B, hidden_dim]
                    self.saved_model_outputs.append(pooled_model_output.detach().cpu())
                else:
                    self.saved_model_outputs.append(model_output_cond.detach().cpu())

            if cfg_mode == "embedding":
                # Apply CFG to model outputs (embeddings) before action decoder
                cfg_model_output = cfg_scale * model_output_cond - (cfg_scale - 1) * model_output_uncond
                pred = self.action_decoder(cfg_model_output, embodiment_id)
                pred_noise_or_x0 = pred[:, -self.action_horizon :]
                
                # Save analysis data for embedding mode
                if self.save_velocity_analysis:
                    self.velocity_analysis_data['pred_noises'].append(pred_noise_or_x0.detach().cpu())
                    
                    print(f"[CFG-Embedding] Step {i}/{num_steps}: pred_noise_norm={pred_noise_or_x0.norm().item():.10f}", flush=True)
                    
                    # Compute cosine similarity with previous noise prediction if available
                    if len(self.velocity_analysis_data['pred_noises']) > 1:
                        prev_noise = self.velocity_analysis_data['pred_noises'][-2]
                        curr_noise = self.velocity_analysis_data['pred_noises'][-1]
                        
                        # Compute cosine similarity per timestep and then average
                        prev_float = prev_noise.to(torch.float32)
                        curr_float = curr_noise.to(torch.float32)
                        
                        cos_sim_per_timestep = F.cosine_similarity(prev_float, curr_float, dim=2, eps=1e-8)
                        cos_sim = cos_sim_per_timestep.mean().item()
                        
                        self.velocity_analysis_data['noise_cos_sims'].append(cos_sim)
                        print(f"[CFG-Embedding] Step {i}/{num_steps}: noise_cos_sim={cos_sim:.10f}", flush=True)
                
            elif cfg_mode == "action":
                # Apply CFG to final actions after action decoder
                pred_cond = self.action_decoder(model_output_cond, embodiment_id)
                pred_uncond = self.action_decoder(model_output_uncond, embodiment_id)
                
                pred_noise_or_x0_cond = pred_cond[:, -self.action_horizon :]
                pred_noise_or_x0_uncond = pred_uncond[:, -self.action_horizon :]
                
                # Apply CFG to noise predictions
                pred_noise_or_x0 = cfg_scale * pred_noise_or_x0_cond - (cfg_scale - 1) * pred_noise_or_x0_uncond
                
                # Save analysis data for action mode
                if self.save_velocity_analysis:
                    self.velocity_analysis_data['pred_noises'].append(pred_noise_or_x0.detach().cpu())
                    self.velocity_analysis_data['pred_noises_uncond'].append(pred_noise_or_x0_uncond.detach().cpu())
                    
                    print(f"[CFG-Action] Step {i}/{num_steps}: pred_noise_norm={pred_noise_or_x0.norm().item():.10f}, pred_noise_uncond_norm={pred_noise_or_x0_uncond.norm().item():.10f}", flush=True)
                    
                    # Compute cosine similarity between conditional and unconditional noise predictions
                    cond_float = pred_noise_or_x0_cond.to(torch.float32)
                    uncond_float = pred_noise_or_x0_uncond.to(torch.float32)

                    cond_uncond_cos_sim_per_timestep = F.cosine_similarity(cond_float, uncond_float, dim=2, eps=1e-8)
                    cond_uncond_cos_sim = cond_uncond_cos_sim_per_timestep.mean().item()
                    self.velocity_analysis_data['cond_uncond_cos_sims'].append(cond_uncond_cos_sim)
                    print(f"[CFG-Action] Step {i}/{num_steps}: cond_uncond_cos_sim={cond_uncond_cos_sim:.10f}", flush=True)
                    
                    # Compute cosine similarity with previous noise prediction if available
                    if len(self.velocity_analysis_data['pred_noises']) > 1:
                        prev_noise = self.velocity_analysis_data['pred_noises'][-2]
                        curr_noise = self.velocity_analysis_data['pred_noises'][-1]
                        
                        prev_float = prev_noise.to(torch.float32)
                        curr_float = curr_noise.to(torch.float32)
                        
                        cos_sim_per_timestep = F.cosine_similarity(prev_float, curr_float, dim=2, eps=1e-8)
                        cos_sim = cos_sim_per_timestep.mean().item()
                        self.velocity_analysis_data['noise_cos_sims'].append(cos_sim)
                        print(f"[CFG-Action] Step {i}/{num_steps}: noise_cos_sim={cos_sim:.10f}", flush=True)
            
            else:
                raise ValueError(f"Invalid cfg_mode: {cfg_mode}. Must be 'action', 'embedding', or None")

            # DDPM update rule
            alpha_prod_t = self.alphas_cumprod[t_idx]
            
            # Add small epsilon to prevent division by zero
            eps = 1e-8
            
            if self.prediction_type == "epsilon":
                # Clamp predicted noise to prevent extreme values
                pred_noise_or_x0 = torch.clamp(pred_noise_or_x0, -10.0, 10.0)
                
                # Predict x0 from noise prediction (with epsilon for numerical stability)
                sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t.clamp(min=eps))
                sqrt_one_minus_alpha_prod_t = torch.sqrt((1 - alpha_prod_t).clamp(min=eps))
                
                pred_x0 = (actions - sqrt_one_minus_alpha_prod_t * pred_noise_or_x0) / sqrt_alpha_prod_t
            elif self.prediction_type == "sample":
                # Direct x0 prediction
                pred_x0 = pred_noise_or_x0
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
            # Clamp predicted x0 to action space (critical for stability!)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Get next timestep
            if i < len(timestep_indices) - 1:
                next_t_idx = timestep_indices[i + 1].item()
                alpha_prod_t_prev = self.alphas_cumprod[next_t_idx]
                
                sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t.clamp(min=eps))
                sqrt_one_minus_alpha_prod_t = torch.sqrt((1 - alpha_prod_t).clamp(min=eps))
                sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev.clamp(min=eps))
                sqrt_one_minus_alpha_prod_t_prev = torch.sqrt((1 - alpha_prod_t_prev).clamp(min=eps))
                
                if self.use_ddim:
                    # DDIM (deterministic): use predicted direction
                    epsilon_t = (actions - sqrt_alpha_prod_t * pred_x0) / sqrt_one_minus_alpha_prod_t
                    actions = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * epsilon_t
                else:
                    # DDPM (stochastic): add random noise
                    noise = torch.randn_like(actions)
                    actions = sqrt_alpha_prod_t_prev * pred_x0 + sqrt_one_minus_alpha_prod_t_prev * noise
                
                # Clamp actions to prevent explosion
                actions = torch.clamp(actions, -10.0, 10.0)
            else:
                # Final step: return predicted x0
                actions = pred_x0
            
        return BatchFeature(data={"action_pred": actions})

    def clear_saved_model_outputs(self):
        """Clear all saved model outputs."""
        self.saved_model_outputs = []
    
    def get_saved_model_outputs(self):
        """Return all saved model outputs as a single tensor."""
        if not self.saved_model_outputs:
            return None
        return torch.cat(self.saved_model_outputs, dim=0)
    
    def clear_velocity_analysis(self):
        """Clear all velocity analysis data."""
        self.velocity_analysis_data = {
            'dt_values': [],
            'pred_noises': [],
            'pred_noises_uncond': [],
            'noise_cos_sims': [],
            'cond_uncond_cos_sims': []
        }
    
    def get_velocity_analysis(self):
        """Return velocity analysis data."""
        return self.velocity_analysis_data
    
    def enable_velocity_analysis(self):
        """Enable velocity tracking for CFG analysis."""
        self.save_velocity_analysis = True
    
    def disable_velocity_analysis(self):
        """Disable velocity tracking for CFG analysis."""
        self.save_velocity_analysis = False

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

