import cv2
import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from line_profiler import profile

from ac_cpc_contrastive_network import ActionConditionedContrastiveNetwork
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length
from sub_models.world_models import WorldModel as BaseWorldModel
from tools import weight_init


class WorldModel(BaseWorldModel):
    def __init__(self, action_dim, config, device):
        super().__init__(action_dim=action_dim, config=config, device=device)
        self.action_dim = action_dim

        contrastive_cfg = config.Models.WorldModel.Contrastive if "Contrastive" in config.Models.WorldModel else {}
        self.use_contrastive = bool(contrastive_cfg["Enable"]) if "Enable" in contrastive_cfg else True
        self.contrastive_steps = int(contrastive_cfg["Steps"]) if "Steps" in contrastive_cfg else 5
        self.contrastive_hidden_size = int(contrastive_cfg["HiddenSize"]) if "HiddenSize" in contrastive_cfg else self.hidden_state_dim
        self.contrastive_out_size = int(contrastive_cfg["OutSize"]) if "OutSize" in contrastive_cfg else self.hidden_state_dim
        self.contrastive_layers = int(contrastive_cfg["Layers"]) if "Layers" in contrastive_cfg else 2
        self.contrastive_loss_scale = float(contrastive_cfg["LossScale"]) if "LossScale" in contrastive_cfg else 1.0
        self.contrastive_exp_lambda = float(contrastive_cfg["ExpLambda"]) if "ExpLambda" in contrastive_cfg else 0.75
        self.contrastive_temperature = float(contrastive_cfg["Temperature"]) if "Temperature" in contrastive_cfg else 0.1
        self.contrastive_max_samples = int(contrastive_cfg["MaxSamples"]) if "MaxSamples" in contrastive_cfg else 1024
        self.contrastive_steps = max(1, self.contrastive_steps)

        if self.use_contrastive:
            input_size = config.Models.WorldModel.Encoder.InputSize
            crop_size = (int(input_size[1]), int(input_size[2]))
            crop_scale = tuple(contrastive_cfg["CropScale"]) if "CropScale" in contrastive_cfg else (0.25, 1.0)
            crop_ratio = tuple(contrastive_cfg["CropRatio"]) if "CropRatio" in contrastive_cfg else (0.9, 1.1)
            self.contrastive_augment = K.RandomResizedCrop(
                size=crop_size,
                scale=crop_scale,
                ratio=crop_ratio,
                p=1.0,
            )
            self.contrastive_network = nn.ModuleList([
                ActionConditionedContrastiveNetwork(
                    feat_size=self.stoch_flattened_dim + self.hidden_state_dim + t * self.action_dim,
                    embed_size=self.stoch_flattened_dim,
                    hidden_size=self.contrastive_hidden_size,
                    out_size=self.contrastive_out_size,
                    num_layers=self.contrastive_layers,
                    act=config.Models.WorldModel.Act,
                    dtype=config.Models.WorldModel.dtype,
                    device=device,
                )
                for t in range(self.contrastive_steps)
            ])
            self.contrastive_network.apply(weight_init)

            contrastive_params = list(self.contrastive_network.parameters())
            if contrastive_params:
                self.optimizer.add_param_group({"params": contrastive_params})
        else:
            self.contrastive_augment = nn.Identity()
            self.contrastive_network = nn.ModuleList()

    def _augment_for_contrastive(self, obs):
        batch_size, seq_len = obs.shape[:2]
        obs_flat = obs.flatten(start_dim=0, end_dim=1).to(torch.float32)
        obs_aug_flat = self.contrastive_augment(obs_flat)
        obs_aug = rearrange(obs_aug_flat, "(B L) C H W -> B L C H W", B=batch_size, L=seq_len)
        return obs_aug.to(dtype=obs.dtype)

    def _build_future_action_condition(self, action_onehot, horizon):
        valid_length = action_onehot.shape[1] - horizon
        action_chunks = []
        for step in range(horizon):
            start = 1 + step
            end = start + valid_length
            action_chunks.append(action_onehot[:, start:end])
        return torch.cat(action_chunks, dim=-1)

    def _subsample_pairs(self, predictor_input, target_input):
        if predictor_input.dim() > 2:
            predictor_input = predictor_input.flatten(start_dim=0, end_dim=1)
        if target_input.dim() > 2:
            target_input = target_input.flatten(start_dim=0, end_dim=1)

        if self.contrastive_max_samples > 0 and predictor_input.shape[0] > self.contrastive_max_samples:
            sample_idx = torch.randperm(predictor_input.shape[0], device=predictor_input.device)[:self.contrastive_max_samples]
            predictor_input = predictor_input[sample_idx]
            target_input = target_input[sample_idx]

        return predictor_input, target_input

    def _cosine_info_nce_loss(self, predictor_output, representation_output):
        predictor_output = F.normalize(predictor_output.float(), dim=-1)
        representation_output = F.normalize(representation_output.float(), dim=-1)

        logits = predictor_output @ representation_output.transpose(0, 1)
        logits = logits / max(self.contrastive_temperature, 1e-6)

        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, targets)
        acc = (logits.argmax(dim=-1) == targets).float().mean()
        return loss, acc

    @profile
    def update(self, obs, action, reward, termination, is_first, global_step, epoch_step, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            obs_hat = self.image_decoder(flattened_sample)

            r_twohot = self.symlog_twohot_loss_func.encode(reward).to(flattened_sample.dtype)
            r_shifted = torch.zeros_like(r_twohot)
            r_shifted[:, 1:, :] = r_twohot[:, :-1, :]
            is_first = is_first.to(device=flattened_sample.device, dtype=flattened_sample.dtype)

            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
                dist_feat = self.sequence_model(flattened_sample, action, temporal_mask, is_first=is_first)
            else:
                mamba_input = self._prepare_mamba_input(flattened_sample, action, r_shifted)
                dist_feat = self.sequence_model(mamba_input, is_first=is_first)
            prior_logits = self.dist_head.forward_prior(dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)

            contrastive_loss = flattened_sample.new_tensor(0.0)
            contrastive_acc = flattened_sample.new_tensor(0.0)

            if self.use_contrastive and len(self.contrastive_network) > 0:
                obs_aug = self._augment_for_contrastive(obs)
                embedding_aug = self.encoder(obs_aug)
                post_logits_aug = self.dist_head.forward_post(embedding_aug)
                sample_aug = self.stright_throught_gradient(post_logits_aug, sample_mode="random_sample")
                flattened_sample_aug = self.flatten_sample(sample_aug)

                prior_feature = torch.cat([prior_flattened_sample, dist_feat], dim=-1)
                action_onehot = F.one_hot(action.long(), self.action_dim).to(prior_feature.dtype)

                max_steps = min(self.contrastive_steps, batch_length)
                contrastive_weight_denom = sum(self.contrastive_exp_lambda ** t for t in range(max_steps))

                for t in range(max_steps):
                    if t == 0:
                        predictor_input = prior_feature
                        target_input = flattened_sample_aug
                    else:
                        action_condition = self._build_future_action_condition(action_onehot, t)
                        predictor_input = torch.cat([prior_feature[:, :-t], action_condition], dim=-1)
                        target_input = flattened_sample_aug[:, t:]

                    predictor_input, target_input = self._subsample_pairs(predictor_input, target_input)
                    predictor_output, representation_output = self.contrastive_network[t](predictor_input, target_input)
                    step_loss, step_acc = self._cosine_info_nce_loss(predictor_output, representation_output)

                    step_weight = (self.contrastive_exp_lambda ** t) / contrastive_weight_denom
                    contrastive_loss = contrastive_loss + self.contrastive_loss_scale * step_weight * step_loss.to(contrastive_loss.dtype)
                    contrastive_acc = contrastive_acc + step_acc.to(contrastive_acc.dtype)

                contrastive_acc = contrastive_acc / max_steps

            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            reconstruction_loss = self.mse_loss_func(obs_hat[:batch_size], obs[:batch_size])
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            total_loss = reconstruction_loss + reward_loss + termination_loss + dynamics_loss + 0.1 * representation_loss + contrastive_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.warmup_scheduler.dampen()

        if (global_step + epoch_step) % self.save_every_steps == 0:
            sample_obs = torch.clamp(obs[:3, 0, :] * 255, 0, 255).permute(0, 2, 3, 1).cpu().detach().float().numpy().astype(np.uint8)
            sample_obs_hat = torch.clamp(obs_hat[:3, 0, :] * 255, 0, 255).permute(0, 2, 3, 1).cpu().detach().float().numpy().astype(np.uint8)

            concatenated_images = []
            for idx in range(3):
                concatenated_image = np.concatenate((sample_obs[idx], sample_obs_hat[idx]), axis=0)
                concatenated_images.append(concatenated_image)

            final_image = np.concatenate(concatenated_images, axis=1)
            height, width, _ = final_image.shape
            scale_factor = 6
            final_image_resized = cv2.resize(final_image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_NEAREST)
            if logger is not None:
                logger.log("Reconstruct/Reconstructed images", [final_image_resized], global_step=global_step)

        return (
            reconstruction_loss.item(),
            reward_loss.item(),
            termination_loss.item(),
            dynamics_loss.item(),
            dynamics_real_kl_div.item(),
            representation_loss.item(),
            representation_real_kl_div.item(),
            total_loss.item(),
            contrastive_loss.item(),
            contrastive_acc.item(),
        )
