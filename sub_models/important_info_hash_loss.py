import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from einops import rearrange # 실시간 이미지 텐서 조립을 위해 추가
import random # 모듈 최상단으로 이동 (4번 지적 반영)

class FastHashBucket:
    """O(N) 병목을 제거하고 완전한 O(1) 연산을 지원하는 리스트 기반 해시 버킷"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []

    def append(self, item):
        if len(self.data) < self.max_size:
            self.data.append(item)
        else:
            # 큐가 꽉 찼을 때 랜덤한 위치를 덮어씌움 (O(1) 유지 및 다양성 확보)
            replace_idx = random.randrange(self.max_size)
            self.data[replace_idx] = item

    def sample_and_remove(self, k):
        if not self.data:
            return []
        k = min(k, len(self.data))
        # 파이썬 list에서의 random.sample은 O(k)로 동작
        indices = random.sample(range(len(self.data)), k)
        samples = [self.data[i] for i in indices]
        
        # O(1) 삭제를 위한 Swap-and-Pop (인덱스가 꼬이지 않도록 내림차순 정렬 후 삭제)
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
            # 훈련이 길어질 경우 float32 정밀도 오버플로우(Truncation) 방지를 위해 float64 사용 (2번 지적 반영)
            self.register_buffer("td_error_mean", torch.tensor(0.0, dtype=torch.float64))
            self.register_buffer("td_error_var", torch.tensor(1.0, dtype=torch.float64))
            self.register_buffer("td_error_count", torch.tensor(1e-4, dtype=torch.float64))

    # [수정] 웰포드 알고리즘에 valid_mask를 인자로 받아 더미 데이터를 통계에서 제외
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

        # 모델 나머지 연산과 호환을 위해 float32로 변환하여 반환
        return self.td_error_mean.to(torch.float32), self.td_error_var.to(torch.float32)

    def _hash_keys(self, latent):
        if latent.numel() == 0:
            return []
        scores = latent.float() @ self.hash_proj.float()
        bits = scores > 0
        bit_values = self.hash_bit_values.to(bits.device)
        keys = (bits.to(torch.int64) * bit_values).sum(dim=-1)
        return keys.detach().cpu().tolist()

    # obs 대신 ReplayBuffer 인덱스 저장으로 변경
    def _update_memory(self, obs, reward, latent, reward_mean, reward_std, td_error=None, indexes=None):
        if not self.enabled or indexes is None:
            return
        if obs.numel() == 0:
            return
            
        # 차원 불일치 버그 픽스 (NxN 오염 및 브로드캐스팅 방지)
        if reward.dim() == 3:
            reward = reward.squeeze(-1)
        if td_error is not None and td_error.dim() == 3:
            td_error = td_error.squeeze(-1)

        # [수정] 시퀀스의 마지막 스텝(미래 가치를 알 수 없는 더미 0.0)을 마스킹
        valid_mask = torch.ones_like(reward, dtype=torch.bool)
        if valid_mask.dim() >= 2:
            valid_mask[..., -1] = False

        if self.store_only_triggered:
            if self.trigger_type == "td_error":
                # td_error가 None일 때의 예외 처리 추가 (forward와 일관성 유지)
                if td_error is None:
                    raise ValueError("TriggerType이 'td_error'일 경우 _update_memory에도 td_error를 전달해야 합니다.")
                # 2번 에러 픽스: 비교 연산 시에도 float32 적용
                td_error_float = td_error.to(torch.float32)
                td_std = torch.sqrt(self.td_error_var.to(torch.float32) + 1e-8)
                store_mask = torch.abs(td_error_float - self.td_error_mean.to(torch.float32)) >= self.sigma_threshold * td_std
            else:
                store_mask = torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std
        else:
            store_mask = torch.ones_like(reward, dtype=torch.bool)

        # [수정] valid_mask를 강제 적용하여 더미 데이터의 저장을 원천 차단
        store_mask = store_mask & valid_mask

        if not torch.any(store_mask):
            return

        indexes_cpu = indexes.detach().cpu()
        reward_cpu = reward.detach().cpu()
        store_mask_cpu = store_mask.detach().cpu()
        
        latent_masked = latent.detach()[store_mask]
        keys = self._hash_keys(latent_masked)
        if not keys:
            return

        index_items = indexes_cpu[store_mask_cpu]
        reward_items = reward_cpu[store_mask_cpu]

        # 이미지 텐서가 아닌 정수 인덱스만 큐에 기록
        for key, index_item, reward_item in zip(keys, index_items, reward_items):
            queue = self.hash_memory.get(key)
            if queue is None:
                # deque 대신 FastHashBucket 사용
                queue = FastHashBucket(max_size=self.max_queue_per_key)
                self.hash_memory[key] = queue
            queue.append((int(index_item), float(reward_item)))

    # 2번 최적화 적용 및 2번 버그(NxN 차원 팽창) 방지를 위한 Squeeze 적용, ReplayBuffer 연동
    def forward(self, obs, latent, logits, reward, encode_fn, reward_mean, reward_std, td_error=None, indexes=None, replay_buffer=None):
        if not self.enabled:
            return latent.new_tensor(0.0)
        if obs.numel() == 0:
            return latent.new_tensor(0.0)
            
        # 차원 불일치 버그 픽스 (NxN 오염 및 브로드캐스팅 방지)
        if reward.dim() == 3:
            reward = reward.squeeze(-1)
        if td_error is not None and td_error.dim() == 3:
            td_error = td_error.squeeze(-1)

        # [수정] 시퀀스의 마지막 스텝(더미 데이터)을 마스킹
        valid_mask = torch.ones_like(reward, dtype=torch.bool)
        if valid_mask.dim() >= 2:
            valid_mask[..., -1] = False

        # config의 TriggerType에 따른 조건부 트리거 마스크 생성
        if self.trigger_type == "td_error":
            if td_error is None:
                raise ValueError("TriggerType이 'td_error'일 경우 forward에 td_error를 반드시 전달해야 합니다.")
            # 분산 계산의 안정성을 위해 float32 변환 후 연산
            td_error_float = td_error.detach().to(torch.float32)
            # [수정] 통계 오염 방지를 위해 valid_mask를 Welford 업데이트에 전달
            td_mean, td_var = self._update_welford(td_error_float, valid_mask)
            td_std = torch.sqrt(td_var + 1e-8)
            trigger_mask = torch.abs(td_error_float - td_mean) >= self.sigma_threshold * td_std
        else:
            trigger_mask = torch.abs(reward - reward_mean) >= self.sigma_threshold * reward_std

        # [수정] valid_mask를 적용하여 마지막 스텝이 트리거되는 논리적 오류 방지
        trigger_mask = trigger_mask & valid_mask

        if not torch.any(trigger_mask) or not self.hash_memory:
            self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, indexes)
            return latent.new_tensor(0.0)

        latent_trigger = latent[trigger_mask]
        reward_trigger = reward[trigger_mask]
        logits_trigger = logits[trigger_mask] # 재계산을 막기 위해 미리 뽑아둔 로짓 활용
        keys = self._hash_keys(latent_trigger.detach())

        past_index_list = []
        past_reward_list = []
        curr_logits_list = [] # obs 대신 logits 수집
        curr_reward_list = []
        pair_count = 0

        for idx, key in enumerate(keys):
            queue = self.hash_memory.get(key)
            if not queue:
                continue

            # O(N) 큐 재생성 병목 완벽 최적화
            # FastHashBucket의 sample_and_remove를 통해 O(k) 시간에 랜덤 샘플링 및 삭제 처리
            available_slots = self.max_pairs_per_batch - pair_count
            k = min(self.max_past_samples, available_slots)
            
            entries = queue.sample_and_remove(k)

            if not entries:
                break

            for index_item, reward_item in entries:
                past_index_list.append(index_item)
                past_reward_list.append(reward_item)
                curr_logits_list.append(logits_trigger[idx]) # 추가 인코딩 없이 로짓 담기
                curr_reward_list.append(reward_trigger[idx])
                pair_count += 1
            
            if pair_count >= self.max_pairs_per_batch:
                break

        if pair_count == 0:
            self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, indexes)
            return latent.new_tensor(0.0)

        if replay_buffer is None:
            raise ValueError("replay_buffer is required to fetch past observations.")

        # 저장해둔 인덱스를 기반으로 버퍼에서 직접 최신화된 과거 이미지 텐서들을 꺼내옴
        if replay_buffer.store_on_gpu:
            past_obs_raw = replay_buffer.obs_buffer[past_index_list]
        else:
            past_obs_raw = torch.from_numpy(replay_buffer.obs_buffer[past_index_list])
            
        past_obs = past_obs_raw.to(device=latent.device, dtype=latent.dtype) / 255.0
        # ReplayBuffer에 담긴 Shape(N, H, W, C)를 Encoder 입력용 Shape(N, 1, C, H, W)로 변환
        past_obs = rearrange(past_obs, "N H W C -> N 1 C H W")
        
        # curr_obs를 다시 CNN에 넣을 필요 없이 모아둔 로짓 바로 사용
        curr_logits = torch.stack(curr_logits_list, dim=0)

        # VRAM 메모리 누수 방지! 연산 그래프가 생성되지 않도록 반드시 no_grad 컨텍스트 안에서 실행 (1번 지적 반영)
        with torch.no_grad():
            # encode_fn은 (latent, logits) 튜플을 반환하여 방 배정용과 비교용을 분리 지원
            past_latent, past_logits = encode_fn(past_obs)
            past_latent, past_logits = past_latent.squeeze(1), past_logits.squeeze(1)

        # 과거 표상은 움직이지 않는 닻(Anchor) 역할을 하도록 역전파 완벽 차단
        past_latent = past_latent.detach()
        past_logits = past_logits.detach()

        # Re-hashing: 최신 인코더에서 추출된 past_latent로 새로운 방 번호(Key) 계산 후 재삽입
        new_keys = self._hash_keys(past_latent)
        for i, new_key in enumerate(new_keys):
            target_queue = self.hash_memory.get(new_key)
            if target_queue is None:
                target_queue = FastHashBucket(max_size=self.max_queue_per_key)
                self.hash_memory[new_key] = target_queue
            # 다시 넣을 때도 이미지 대신 인덱스 보관
            target_queue.append((past_index_list[i], float(past_reward_list[i])))

        # Squeeze가 적용되어 curr_reward는 (N,), past_reward도 (N,)이 됨
        curr_reward = torch.stack(curr_reward_list, dim=0)
        past_reward = torch.tensor(past_reward_list, device=latent.device, dtype=curr_reward.dtype)

        reward_diff = torch.abs(curr_reward - past_reward)
        # 노이즈가 제거된 logits를 사용하여 코사인 유사도 정밀 계산
        cosine_sim = F.cosine_similarity(curr_logits, past_logits, dim=-1, eps=1e-8)
        
        # Margin 기반 척력 설계: 보상이 다를 때 표상을 무한히(-1) 밀어내지 않고 직교(0)까지만 밀어냄. 
        # 보상이 같으면(reward_diff=0) Loss는 0이 되어, 다른 모듈의 구조화에 간섭하지 않음.
        margin = 0.0
        loss = (reward_diff * F.relu(cosine_sim - margin)).mean() * self.loss_scale

        self._update_memory(obs, reward, latent, reward_mean, reward_std, td_error, indexes)
        return loss