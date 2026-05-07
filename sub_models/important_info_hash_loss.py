import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ImportantInfoHashLoss(nn.Module):
    def __init__(self, latent_dim, config):
        super().__init__()
        cfg = config or {}
        self.enabled = bool(cfg.get("Enable", False))
        
        # 2 sigma 임계값으로 대체 적용
        self.sigma_threshold = 2.0
        
        self.hash_bits = int(cfg.get("HashBits", 12))
        self.max_queue_per_key = int(cfg.get("MaxQueuePerKey", 8))
        self.max_past_samples = int(cfg.get("MaxPastSamples", 4))
        self.max_pairs_per_batch = int(cfg.get("MaxPairsPerBatch", 256))
        self.loss_scale = float(cfg.get("LossScale", 1.0))
        self.store_only_triggered = bool(cfg.get("StoreOnlyTriggered", True))

        if self.hash_bits < 1 or self.hash_bits > 62:
            raise ValueError(f"HashBits must be in [1, 62], got {self.hash_bits}.")
        self.max_queue_per_key = max(1, self.max_queue_per_key)
        self.max_past_samples = max(1, self.max_past_samples)
        self.max_pairs_per_batch = max(1, self.max_pairs_per_batch)

        proj = torch.randn(latent_dim, self.hash_bits, dtype=torch.float32)
        self.register_buffer("hash_proj", proj, persistent=False)
        bit_values = 2 ** torch.arange(self.hash_bits, dtype=torch.int64)
        self.register_buffer("hash_bit_values", bit_values, persistent=False)

        self.hash_memory = {}

    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return []
        scores = latent.float() @ self.hash_proj.float()
        bits = scores > 0
        bit_values = self.hash_bit_values.to(bits.device)
        keys = (bits.to(torch.int64) * bit_values).sum(dim=-1)
        return keys.detach().cpu().tolist()

    def _update_memory(self, obs, reward, latent, reward_mean, reward_std):
        if not self.enabled:
            return
        if obs.numel() == 0:
            return

        if self.store_only_triggered:
            store_mask = torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std
        else:
            store_mask = torch.ones_like(reward, dtype=torch.bool)

        if not torch.any(store_mask):
            return

        obs_uint8 = torch.clamp(obs.detach() * 255.0, 0, 255).to(torch.uint8).cpu()
        reward_cpu = reward.detach().cpu()
        store_mask_cpu = store_mask.detach().cpu()
        latent_masked = latent.detach()[store_mask]
        keys = self._hash_keys(latent_masked)
        if not keys:
            return

        obs_items = obs_uint8[store_mask_cpu]
        reward_items = reward_cpu[store_mask_cpu]

        for key, obs_item, reward_item in zip(keys, obs_items, reward_items):
            queue = self.hash_memory.get(key)
            if queue is None:
                queue = deque(maxlen=self.max_queue_per_key)
                self.hash_memory[key] = queue
            queue.append((obs_item, float(reward_item)))

    def forward(self, obs, latent, reward, encode_fn, reward_mean, reward_std):
        if not self.enabled:
            return latent.new_tensor(0.0)
        if obs.numel() == 0:
            return latent.new_tensor(0.0)

        trigger_mask = torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std
        if not torch.any(trigger_mask) or not self.hash_memory:
            self._update_memory(obs, reward, latent, reward_mean, reward_std)
            return latent.new_tensor(0.0)

        latent_trigger = latent[trigger_mask]
        reward_trigger = reward[trigger_mask]
        keys = self._hash_keys(latent_trigger.detach())

        past_obs_list = []
        past_reward_list = []
        curr_latent_list = []
        curr_reward_list = []
        pair_count = 0

        for idx, key in enumerate(keys):
            queue = self.hash_memory.get(key)
            if not queue:
                continue

            entries = list(queue)[-self.max_past_samples:]
            for obs_item, reward_item in entries:
                past_obs_list.append(obs_item)
                past_reward_list.append(reward_item)
                curr_latent_list.append(latent_trigger[idx])
                curr_reward_list.append(reward_trigger[idx])
                pair_count += 1
                if pair_count >= self.max_pairs_per_batch:
                    break
            if pair_count >= self.max_pairs_per_batch:
                break

        if pair_count == 0:
            self._update_memory(obs, reward, latent, reward_mean, reward_std)
            return latent.new_tensor(0.0)

        past_obs = torch.stack(past_obs_list, dim=0)
        past_obs = past_obs.to(device=latent.device, dtype=latent.dtype) / 255.0
        past_obs = past_obs.unsqueeze(1)
        past_latent = encode_fn(past_obs).squeeze(1)

        curr_latent = torch.stack(curr_latent_list, dim=0)
        curr_reward = torch.stack(curr_reward_list, dim=0)
        past_reward = torch.tensor(past_reward_list, device=latent.device, dtype=curr_reward.dtype)

        reward_diff = torch.abs(curr_reward - past_reward)
        cosine_sim = F.cosine_similarity(curr_latent, past_latent, dim=-1, eps=1e-8)
        loss = (reward_diff * cosine_sim).mean() * self.loss_scale

        self._update_memory(obs, reward, latent, reward_mean, reward_std)
        return loss
