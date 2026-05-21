import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from einops import rearrange
import random

class FastHashBucket:
    """O(N) 병목을 제거하고 완전한 O(1) 연산을 지원하는 리스트 기반 해시 버킷"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []

    def append(self, item):
        if len(self.data) < self.max_size:
            self.data.append(item)
        else:
            replace_idx = random.randrange(self.max_size)
            self.data[replace_idx] = item

    def sample_and_remove(self, k):
        if not self.data:
            return []
        k = min(k, len(self.data))
        indices = random.sample(range(len(self.data)), k)
        samples = [self.data[i] for i in indices]
        
        indices.sort(reverse=True)
        for idx in indices:
            self.data[idx] = self.data[-1]
            self.data.pop()
            
        return samples

    def __len__(self):
        return len(self.data)


class ImportantInfoHashLoss(nn.Module):
    def __init__(self, latent_dim, config):
        super().__init__()
        cfg = config or {}
        self.enabled = bool(cfg.get("Enable", False))
        
        # TriggerType 설정 (reward 또는 td_error)
        self.trigger_type = str(cfg.get("TriggerType", "reward")).lower()
        
        # Contrastive 거리에 사용할 Metric 설정
        self.diff_type = str(cfg.get("DiffType", "reward")).lower()
        if self.diff_type not in ["reward", "td_error", "value", "aux_value"]:
            raise ValueError(f"DiffType must be 'reward', 'td_error', 'value', or 'aux_value', got {self.diff_type}")
            
        # Stateless Value Distillation을 위한 보조 네트워크 (Mamba State 없이 오직 latent 표상으로만 가치 매핑)
        self.aux_value_net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
        
        # RewardThreshold 대신 SigmaThreshold를 설정에서 읽어옴 (기본값 2.0)
        self.sigma_threshold = float(cfg.get("SigmaThreshold", 2.0))
        
        self.hash_bits = int(cfg.get("HashBits", 12))
        self.max_queue_per_key = int(cfg.get("MaxQueuePerKey", 100000))
        self.max_past_samples = int(cfg.get("MaxPastSamples", 4))
        self.max_pairs_per_batch = int(cfg.get("MaxPairsPerBatch", 256))
        self.loss_scale = float(cfg.get("LossScale", 1.0))
        self.store_only_triggered = bool(cfg.get("StoreOnlyTriggered", False))

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
        
        # TD-Error의 웰포드 알고리즘 연산을 위한 버퍼 등록
        if self.trigger_type == "td_error":
            self.register_buffer("td_error_mean", torch.tensor(0.0, dtype=torch.float64))
            self.register_buffer("td_error_var", torch.tensor(1.0, dtype=torch.float64))
            self.register_buffer("td_error_count", torch.tensor(1e-4, dtype=torch.float64))

    def _update_welford(self, td_error, valid_mask):
        valid_td = td_error[valid_mask].to(torch.float64)
        
        if valid_td.numel() == 0:
            return self.td_error_mean.to(torch.float32), self.td_error_var.to(torch.float32)

        batch_mean = torch.mean(valid_td)
        batch_var = torch.var(valid_td, unbiased=False)
        batch_count = torch.tensor(valid_td.numel(), dtype=torch.float64, device=td_error.device)

        delta = batch_mean - self.td_error_mean
        tot_count = self.td_error_count + batch_count

        new_mean = self.td_error_mean + delta * batch_count / tot_count
        m_a = self.td_error_var * self.td_error_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.td_error_count * batch_count / tot_count
        new_var = M2 / tot_count

        self.td_error_mean.copy_(new_mean)
        self.td_error_var.copy_(new_var)
        self.td_error_count.copy_(tot_count)

        return self.td_error_mean.to(torch.float32), self.td_error_var.to(torch.float32)

    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return []
        scores = latent.float() @ self.hash_proj.float()
        bits = scores > 0
        bit_values = self.hash_bit_values.to(bits.device)
        keys = (bits.to(torch.int64) * bit_values).sum(dim=-1)
        return keys.detach().cpu().tolist()

    def _update_memory(self, obs, reward, latent, reward_mean, reward_std, td_error=None, value=None, indexes=None):
        if not self.enabled or indexes is None:
            return
        if obs.numel() == 0:
            return
            
        if reward.dim() == 3:
            reward = reward.squeeze(-1)
        if td_error is not None and td_error.dim() == 3:
            td_error = td_error.squeeze(-1)
        if value is not None and value.dim() == 3:
            value = value.squeeze(-1)

        # [수정] 시점 정렬 교정으로 인해 V0 및 마스크를 알 수 없는 처음(0)과 마지막(-1) 스텝을 모두 마스킹 처리
        valid_mask = torch.ones_like(reward, dtype=torch.bool)
        if valid_mask.dim() >= 2:
            valid_mask[..., 0] = False
            valid_mask[..., -1] = False

        if self.store_only_triggered:
            if self.trigger_type == "td_error":
                if td_error is None:
                    raise ValueError("TriggerType이 'td_error'일 경우 _update_memory에도 td_error를 전달해야 합니다.")
                td_error_float = td_error.to(torch.float32)
                td_std = torch.sqrt(self.td_error_var.to(torch.float32) + 1e-8)
                store_mask = torch.abs(td_error_float - self.td_error_mean.to(torch.float32)) >= self.sigma_threshold * td_std
            else:
                store_mask = torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std
        else:
            store_mask = torch.ones_like(reward, dtype=torch.bool)

        store_mask = store_mask & valid_mask

        if not torch.any(store_mask):
            return

        if self.diff_type == "td_error":
            metric_to_store = td_error
        elif self.diff_type in ["value", "aux_value"]:
            metric_to_store = value if value is not None else torch.zeros_like(reward)
        else:
            metric_to_store = reward

        indexes_cpu = indexes.detach().cpu()
        metric_cpu = metric_to_store.detach().cpu()
        store_mask_cpu = store_mask.detach().cpu()
        
        latent_masked = latent.detach()[store_mask]
        keys = self._hash_keys(latent_masked)
        if not keys:
            return

        index_items = indexes_cpu[store_mask_cpu]
        metric_items = metric_cpu[store_mask_cpu]

        for key, index_item, metric_item in zip(keys, index_items, metric_items):
            queue = self.hash_memory.get(key)
            if queue is None:
                queue = FastHashBucket(max_size=self.max_queue_per_key)
                self.hash_memory[key] = queue
            queue.append((int(index_item), float(metric_item)))

    def forward(self, obs, latent, logits, reward, encode_fn, reward_mean, reward_std, td_error=None, value=None, indexes=None, replay_buffer=None):
        if not self.enabled:
            return latent.new_tensor(0.0)
        if obs.numel() == 0:
            return latent.new_tensor(0.0)
            
        if self.diff_type == "value" and value is None:
            raise ValueError("DiffType이 'value'일 경우 forward에 value를 반드시 전달해야 합니다.")
        if self.diff_type == "td_error" and td_error is None:
            raise ValueError("DiffType이 'td_error'일 경우 forward에 td_error를 반드시 전달해야 합니다.")
        if self.diff_type == "aux_value" and value is None:
            raise ValueError("DiffType이 'aux_value'일 경우 distillation 학습을 위해 forward에 value를 반드시 전달해야 합니다.")
            
        if reward.dim() == 3:
            reward = reward.squeeze(-1)
        if td_error is not None and td_error.dim() == 3:
            td_error = td_error.squeeze(-1)
        if value is not None and value.dim() == 3:
            value = value.squeeze(-1)

        # [수정] 시점 정렬 교정으로 인해 정렬 영역을 벗어난 처음(0)과 마지막(-1) 스텝을 일괄 유효 통계에서 차단
        valid_mask = torch.ones_like(reward, dtype=torch.bool)
        if valid_mask.dim() >= 2:
            valid_mask[..., 0] = False
            valid_mask[..., -1] = False

        if value is not None:
            aux_value_all = self.aux_value_net(latent).squeeze(-1)
            distill_loss = F.mse_loss(aux_value_all[valid_mask], value[valid_mask].detach())
        else:
            distill_loss = latent.new_tensor(0.0)

        if self.trigger_type == "td_error":
            if td_error is None:
                raise ValueError("TriggerType이 'td_error'일 경우 forward에 td_error를 반드시 전달해야 합니다.")
            td_error_float = td_error.detach().to(torch.float32)
            td_mean, td_var = self._update_welford(td_error_float, valid_mask)
            td_std = torch.sqrt(td_var + 1e-8)
            trigger_mask = torch.abs(td_error_float - td_mean) >= self.sigma_threshold * td_std
        else:
            trigger_mask = torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std

        trigger_mask = trigger_mask & valid_mask

        if not torch.any(trigger_mask) or not self.hash_memory:
            self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, value, indexes)
            return distill_loss

        latent_trigger = latent[trigger_mask]
        logits_trigger = logits[trigger_mask]
        
        if self.diff_type == "td_error":
            metric_trigger = td_error[trigger_mask]
        elif self.diff_type == "value":
            metric_trigger = value[trigger_mask]
        elif self.diff_type == "aux_value":
            metric_trigger = aux_value_all[trigger_mask]
        else:
            metric_trigger = reward[trigger_mask]
            
        keys = self._hash_keys(latent_trigger.detach())

        past_index_list = []
        past_metric_list = []
        curr_logits_list = []
        curr_metric_list = []
        pair_count = 0

        for idx, key in enumerate(keys):
            queue = self.hash_memory.get(key)
            if not queue:
                continue

            available_slots = self.max_pairs_per_batch - pair_count
            k = min(self.max_past_samples, available_slots)
            entries = queue.sample_and_remove(k)

            if not entries:
                break

            for index_item, metric_item in entries:
                past_index_list.append(index_item)
                past_metric_list.append(metric_item)
                curr_logits_list.append(logits_trigger[idx])
                curr_metric_list.append(metric_trigger[idx])
                pair_count += 1
            
            if pair_count >= self.max_pairs_per_batch:
                break

        if pair_count == 0:
            self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, value, indexes)
            return distill_loss

        if replay_buffer is None:
            raise ValueError("replay_buffer is required to fetch past observations.")

        if replay_buffer.store_on_gpu:
            past_obs_raw = replay_buffer.obs_buffer[past_index_list]
        else:
            past_obs_raw = torch.from_numpy(replay_buffer.obs_buffer[past_index_list])
            
        past_obs = past_obs_raw.to(device=latent.device, dtype=latent.dtype) / 255.0
        past_obs = rearrange(past_obs, "N H W C -> N 1 C H W")
        
        curr_logits = torch.stack(curr_logits_list, dim=0)

        with torch.no_grad():
            past_latent, past_logits = encode_fn(past_obs)
            past_latent, past_logits = past_latent.squeeze(1), past_logits.squeeze(1)

        past_latent = past_latent.detach()
        past_logits = past_logits.detach()

        new_keys = self._hash_keys(past_latent)
        for i, new_key in enumerate(new_keys):
            target_queue = self.hash_memory.get(new_key)
            if target_queue is None:
                target_queue = FastHashBucket(max_size=self.max_queue_per_key)
                self.hash_memory[new_key] = target_queue
            target_queue.append((past_index_list[i], float(past_metric_list[i])))

        curr_metric = torch.stack(curr_metric_list, dim=0)

        if self.diff_type == "aux_value":
            past_aux_pred = self.aux_value_net(past_latent).squeeze(-1)
            metric_diff = torch.abs(curr_metric.detach() - past_aux_pred.detach())
        else:
            past_metric = torch.tensor(past_metric_list, device=latent.device, dtype=curr_metric.dtype)
            metric_diff = torch.abs(curr_metric - past_metric)
        
        cosine_sim = F.cosine_similarity(curr_logits, past_logits, dim=-1, eps=1e-8)
        
        margin = 0.0
        loss = (metric_diff * F.relu(cosine_sim - margin)).mean() * self.loss_scale + distill_loss

        self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, value, indexes)
        return loss
