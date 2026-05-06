import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, config, device="cuda") -> None:
        self.store_on_gpu = config.BasicSettings.ReplayBufferOnGPU
        max_length = config.JointTrainAgent.BufferMaxLength
        obs_shape = (config.BasicSettings.ImageSize, config.BasicSettings.ImageSize, config.BasicSettings.ImageChannel)
        self.device = device

        if self.store_on_gpu:
            self.obs_buffer = torch.empty((max_length, *obs_shape), dtype=torch.uint8, device=device, requires_grad=False)
            self.action_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.is_first_buffer = torch.empty((max_length), dtype=torch.float32, device=device, requires_grad=False)
            self.sampled_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
            self.imagined_counter = torch.zeros((max_length), dtype=torch.int32, device=device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length), dtype=np.float32)
            self.reward_buffer = np.empty((max_length), dtype=np.float32)
            self.termination_buffer = np.empty((max_length), dtype=np.float32)
            self.is_first_buffer = np.empty((max_length), dtype=np.float32)
            self.sampled_counter = np.zeros((max_length), dtype=np.int32)
            self.imagined_counter = np.zeros((max_length), dtype=np.int32)

        self.length = 0
        self.last_pointer = -1
        self.max_length = max_length
        self.world_model_warmup_length = config.JointTrainAgent.WorldModelWarmUp
        self.behaviour_warmup_length = config.JointTrainAgent.BehaviourWarmUp
        self.tau = config.JointTrainAgent.Tau
        self.imagination_tau = config.JointTrainAgent.ImaginationTau
        self.alpha = config.JointTrainAgent.Alpha
        self.beta = config.JointTrainAgent.Beta
        self.batch_scale_factor = config.JointTrainAgent.ImagineBatchSize / config.JointTrainAgent.BatchSize
        
        # [추가] configure.yaml에서 데모 보존 비율을 읽어옵니다. (설정이 없으면 기본값 0.25 사용)
        self.demo_ratio = getattr(config.Demonstration, 'DemoRatio', 0.25) if hasattr(config, 'Demonstration') else 0.25

    def ready(self, model_name='world_model'):
        return self.length  > self.world_model_warmup_length if model_name == 'world_model' else self.length  > self.behaviour_warmup_length

    @torch.no_grad()
    def sample(self, batch_size, batch_length, imagine=False):
        if self.store_on_gpu:
            obs_list, action_list, reward_list, termination_list, is_first_list = [], [], [], [], []
            valid_length = self.length + 1 - batch_length
            counts = self.sampled_counter[:valid_length]
            imagine_counts = self.imagined_counter[:valid_length] / self.batch_scale_factor
            
            # 1. 확률 분포 계산
            if imagine:
                linear_penalty = torch.maximum(torch.zeros_like(counts), counts - imagine_counts)
                score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                score = score / self.imagination_tau
                probabilities = torch.softmax(score, dim=0)
            else:
                logits = -counts / self.tau
                probabilities = torch.exp(logits) / torch.sum(torch.exp(logits))

            # 2. 혼합 배치 샘플링
            demo_size = getattr(self, 'protect_size', 0)
            demo_valid_size = demo_size - batch_length if demo_size > batch_length else 0

            if demo_valid_size > 0 and valid_length > demo_valid_size:
                demo_prob_mass = probabilities[:demo_valid_size].sum().item()
                # [수정] yaml에서 불러온 self.demo_ratio 값을 사용합니다.
                target_demo_ratio = max(self.demo_ratio, demo_prob_mass)
                
                num_demo_samples = int(batch_size * target_demo_ratio)
                num_agent_samples = batch_size - num_demo_samples
                
                if num_demo_samples > 0:
                    prob_demo = probabilities[:demo_valid_size] / probabilities[:demo_valid_size].sum()
                    demo_indexes = torch.multinomial(prob_demo, num_demo_samples, replacement=True)
                else:
                    demo_indexes = torch.empty(0, dtype=torch.long, device=self.device)
                    
                if num_agent_samples > 0:
                    prob_agent = probabilities[demo_valid_size:] / probabilities[demo_valid_size:].sum()
                    agent_indexes = torch.multinomial(prob_agent, num_agent_samples, replacement=True) + demo_valid_size
                else:
                    agent_indexes = torch.empty(0, dtype=torch.long, device=self.device)
                    
                start_indexes = torch.cat([demo_indexes, agent_indexes])
                start_indexes = start_indexes[torch.randperm(batch_size, device=self.device)]
            else:
                start_indexes = torch.multinomial(probabilities, batch_size, replacement=True)

            # 3. 데이터 추출
            if not imagine:
                self.sampled_counter[start_indexes] += 1
            else:
                self.imagined_counter[start_indexes] += 1

            indexes = start_indexes.unsqueeze(-1).to(self.device) + torch.arange(batch_length, device=self.device)
            
            obs_list.append(self.obs_buffer[indexes])
            action_list.append(self.action_buffer[indexes])
            reward_list.append(self.reward_buffer[indexes])
            termination_list.append(self.termination_buffer[indexes])
            is_first_list.append(self.is_first_buffer[indexes])

            obs = torch.cat(obs_list, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action_list, dim=0)
            reward = torch.cat(reward_list, dim=0)
            termination = torch.cat(termination_list, dim=0)
            is_first = torch.cat(is_first_list, dim=0)
            
        else:
            obs_list, action_list, reward_list, termination_list, is_first_list = [], [], [], [], []

            if batch_size > 0:
                valid_length = self.length + 1 - batch_length
                counts = self.sampled_counter[:valid_length]
                imagine_counts = self.imagined_counter[:valid_length] / self.batch_scale_factor

                # 1. 확률 분포 계산
                if imagine:
                    linear_penalty = np.maximum(np.zeros_like(counts), counts - imagine_counts)
                    score = counts - self.alpha * imagine_counts - self.beta * linear_penalty
                    score /= self.imagination_tau
                else:
                    score = -counts / self.tau

                exp_score = np.exp(score - np.max(score))
                probabilities = exp_score / np.sum(exp_score)

                # 2. 혼합 배치 샘플링
                demo_size = getattr(self, 'protect_size', 0)
                demo_valid_size = demo_size - batch_length if demo_size > batch_length else 0

                if demo_valid_size > 0 and valid_length > demo_valid_size:
                    demo_prob_mass = probabilities[:demo_valid_size].sum()
                    # [수정] yaml에서 불러온 self.demo_ratio 값을 사용합니다.
                    target_demo_ratio = max(self.demo_ratio, demo_prob_mass)
                    
                    num_demo_samples = int(batch_size * target_demo_ratio)
                    num_agent_samples = batch_size - num_demo_samples
                    
                    if num_demo_samples > 0:
                        prob_demo = probabilities[:demo_valid_size] / probabilities[:demo_valid_size].sum()
                        replace_demo = num_demo_samples > demo_valid_size
                        demo_indexes = np.random.choice(demo_valid_size, size=num_demo_samples, replace=replace_demo, p=prob_demo)
                    else:
                        demo_indexes = np.array([], dtype=np.int64)
                        
                    if num_agent_samples > 0:
                        prob_agent = probabilities[demo_valid_size:] / probabilities[demo_valid_size:].sum()
                        replace_agent = num_agent_samples > len(prob_agent)
                        agent_indexes = np.random.choice(len(prob_agent), size=num_agent_samples, replace=replace_agent, p=prob_agent) + demo_valid_size
                    else:
                        agent_indexes = np.array([], dtype=np.int64)
                        
                    start_indexes = np.concatenate([demo_indexes, agent_indexes])
                    np.random.shuffle(start_indexes)
                else:
                    replace_all = batch_size > len(probabilities)
                    start_indexes = np.random.choice(len(probabilities), size=(batch_size,), replace=replace_all, p=probabilities)

                # 3. 데이터 추출
                if not imagine:
                    self.sampled_counter[start_indexes] += 1
                else:
                    self.imagined_counter[start_indexes] += 1 

                indexes = start_indexes[:, np.newaxis] + np.arange(batch_length)

                obs_seq = self.obs_buffer[indexes]
                action_seq = self.action_buffer[indexes]
                reward_seq = self.reward_buffer[indexes]
                termination_seq = self.termination_buffer[indexes]
                is_first_seq = self.is_first_buffer[indexes]

                obs_seq = torch.from_numpy(obs_seq).float().to(self.device) / 255
                obs_seq = rearrange(obs_seq, "B T H W C -> B T C H W")
                action_seq = torch.from_numpy(action_seq).to(self.device)
                reward_seq = torch.from_numpy(reward_seq).to(self.device)
                termination_seq = torch.from_numpy(termination_seq).to(self.device)
                is_first_seq = torch.from_numpy(is_first_seq).to(self.device)

                obs_list.append(obs_seq)
                action_list.append(action_seq)
                reward_list.append(reward_seq)
                termination_list.append(termination_seq)
                is_first_list.append(is_first_seq)

            obs = torch.cat(obs_list, dim=0) if obs_list else torch.empty(0, device=self.device)
            action = torch.cat(action_list, dim=0) if action_list else torch.empty(0, device=self.device)
            reward = torch.cat(reward_list, dim=0) if reward_list else torch.empty(0, device=self.device)
            termination = torch.cat(termination_list, dim=0) if termination_list else torch.empty(0, device=self.device)
            is_first = torch.cat(is_first_list, dim=0) if is_first_list else torch.empty(0, device=self.device)

        return obs, action, reward, termination, is_first

    def append(self, obs, action, reward, termination, is_first):
        self.last_pointer = (self.last_pointer + 1) % (self.max_length)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.tensor(action, device=self.device)
            self.reward_buffer[self.last_pointer] = torch.tensor(reward, device=self.device)
            self.termination_buffer[self.last_pointer] = torch.tensor(termination, device=self.device)
            self.is_first_buffer[self.last_pointer] = torch.tensor(is_first, device=self.device)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination
            self.is_first_buffer[self.last_pointer] = is_first

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length
