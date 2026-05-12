import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class ImportantInfoHashLoss(nn.Module):
    def __init__(self, latent_dim, config):
        super().__init__()
        cfg = config or {}
        self.enabled = bool(cfg.get("Enable", False))
        
        # [수정] RewardThreshold 대신 SigmaThreshold를 설정에서 읽어옴 (기본값 2.0)
        self.sigma_threshold = float(cfg.get("SigmaThreshold", 2.0))
        
        self.hash_bits = int(cfg.get("HashBits", 12))
        self.max_queue_per_key = int(cfg.get("MaxQueuePerKey", 100000))
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

    # [수정] 2번 최적화 적용: 이미 계산된 연속 확률(logits)을 직접 인자로 넘겨받음
    def forward(self, obs, latent, logits, reward, encode_fn, reward_mean, reward_std):
        import random  # 무작위 샘플링을 위해 추가

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
        logits_trigger = logits[trigger_mask] # [수정] 재계산을 막기 위해 미리 뽑아둔 로짓 활용
        keys = self._hash_keys(latent_trigger.detach())

        past_obs_list = []
        past_reward_list = []
        curr_logits_list = [] # [수정] obs 대신 logits 수집
        curr_reward_list = []
        pair_count = 0

        for idx, key in enumerate(keys):
            queue = self.hash_memory.get(key)
            if not queue:
                continue

            # 저장된 데이터가 뽑으려는 개수보다 많으면 '무작위'로 추출, 적으면 전부 사용
            queue_list = list(queue)
            if len(queue_list) > self.max_past_samples:
                entries = random.sample(queue_list, self.max_past_samples)
            else:
                entries = queue_list

            # [수정] 1번 버그 픽스: 남은 슬롯 수만큼만 자르기 (데이터 영구 증발 방지)
            available_slots = self.max_pairs_per_batch - pair_count
            if len(entries) > available_slots:
                entries = entries[:available_slots]

            if not entries:
                break

            # [추가] Re-hashing(방 이동)을 위해 선택된 샘플만 기존 큐에서 안전하게 제거
            entries_ids = set(id(e) for e in entries)
            new_queue = deque([e for e in queue if id(e) not in entries_ids], maxlen=self.max_queue_per_key)
            self.hash_memory[key] = new_queue

            for obs_item, reward_item in entries:
                past_obs_list.append(obs_item)
                past_reward_list.append(reward_item)
                curr_logits_list.append(logits_trigger[idx]) # [수정] 추가 인코딩 없이 로짓 담기
                curr_reward_list.append(reward_trigger[idx])
                pair_count += 1
            
            # 여기서 break 되어도, 위에서 Slicing을 했기 때문에 삭제된 데이터 유실이 없음
            if pair_count >= self.max_pairs_per_batch:
                break

        if pair_count == 0:
            self._update_memory(obs, reward, latent, reward_mean, reward_std)
            return latent.new_tensor(0.0)

        past_obs = torch.stack(past_obs_list, dim=0)
        past_obs = past_obs.to(device=latent.device, dtype=latent.dtype) / 255.0
        past_obs = past_obs.unsqueeze(1)
        
        # [수정] curr_obs를 다시 CNN에 넣을 필요 없이 모아둔 로짓 바로 사용
        curr_logits = torch.stack(curr_logits_list, dim=0)

        # [수정] encode_fn은 (latent, logits) 튜플을 반환하여 방 배정용과 비교용을 분리 지원
        past_latent, past_logits = encode_fn(past_obs)
        past_latent, past_logits = past_latent.squeeze(1), past_logits.squeeze(1)

        # [수정] 과거 표상은 움직이지 않는 닻(Anchor) 역할을 하도록 역전파 완벽 차단
        past_latent = past_latent.detach()
        past_logits = past_logits.detach()

        # [추가] Re-hashing: 최신 인코더에서 추출된 past_latent로 새로운 방 번호(Key) 계산 후 재삽입
        new_keys = self._hash_keys(past_latent)
        for i, new_key in enumerate(new_keys):
            target_queue = self.hash_memory.get(new_key)
            if target_queue is None:
                target_queue = deque(maxlen=self.max_queue_per_key)
                self.hash_memory[new_key] = target_queue
            target_queue.append((past_obs_list[i], float(past_reward_list[i])))

        curr_reward = torch.stack(curr_reward_list, dim=0)
        past_reward = torch.tensor(past_reward_list, device=latent.device, dtype=curr_reward.dtype)

        reward_diff = torch.abs(curr_reward - past_reward)
        # [수정] 노이즈가 제거된 logits를 사용하여 코사인 유사도 정밀 계산
        cosine_sim = F.cosine_similarity(curr_logits, past_logits, dim=-1, eps=1e-8)
        loss = (reward_diff * cosine_sim).mean() * self.loss_scale

        self._update_memory(obs, reward, latent, reward_mean, reward_std)
        return loss