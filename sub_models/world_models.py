import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch.distributions import OneHotCategorical, Normal
# import torchvision.transforms as T
import kornia.augmentation as K
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from sub_models.laprop import LaProp
from pytorch_warmup import LinearWarmup
# from nfnets import AGC

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from sub_models.contrastive_network import ActionConditionedContrastiveNetwork
from sub_models.transformer_model import StochasticTransformerKVCache
from mamba_ssm import MambaWrapperModel, MambaConfig, InferenceParams, update_graph_cache
from mamba_ssm.modules import mamba3 as mamba3_module
import agents
from line_profiler import profile
from torch.distributions.independent import Independent
import numpy as np
from tools import weight_init
import cv2

    
class Encoder(nn.Module):
    def __init__(self, depth=128, mults=(1, 2, 4, 2), norm='rms', act='SiLU', kernel=4, padding='same', first_stride=True, input_size=(3, 64, 64), dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)
        self.depths = [depth * mult for mult in mults]
        self.kernel = kernel
        self.stride = 2
        self.padding = (kernel - 1) // 2 if padding == 'same' else padding
        
        backbone = []
        current_channels, current_height, current_width = input_size

        # Define convolutional layers for image inputs
        for i, depth in enumerate(self.depths):
            stride = 1 if i == 0 and first_stride else self.stride
            conv = nn.Conv2d(in_channels=current_channels, out_channels=depth, kernel_size=kernel, stride=stride, padding=self.padding, dtype=dtype, device=device)
            backbone.append(conv)
            backbone.append(nn.BatchNorm2d(depth, dtype=dtype, device=device))
            backbone.append(act())
            
            current_height, current_width = self._compute_output_dim(current_height, current_width, kernel, stride, self.padding)
            current_channels = depth

        self.backbone = nn.Sequential(*backbone)
        self.backbone.apply(weight_init)
        self.last_channels = self.depths[-1]
        self.output_dim = (self.last_channels, current_height, current_width)
        self.output_flatten_dim = self.last_channels * current_height * current_width

    def _compute_output_dim(self, height, width, kernel_size, stride, padding):
        new_height = (height - kernel_size + 2 * padding) // stride + 1
        new_width = (width - kernel_size + 2 * padding) // stride + 1
        return new_height, new_width

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x

    

class Decoder(nn.Module):
    def __init__(self, stoch_dim, depth=128, mults=(1, 2, 4, 2), norm='rms', act='SiLU', kernel=4, padding='same', first_stride=True, last_output_dim=(256, 4, 4),input_size=(3, 64, 64), cnn_sigmoid=False, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)
        self.depths = [depth * mult for mult in mults]
        self.kernel = kernel
        self.stride = 2
        self.padding = (kernel - 1) // 2 if padding == 'same' else padding
        self.output_padding = self.stride //2 if padding == 'same' else 0
        self._cnn_sigmoid = cnn_sigmoid


        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_output_dim[0] * last_output_dim[1] * last_output_dim[2], bias=True, dtype=dtype, device=device))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_output_dim[0], H=last_output_dim[1]))
        backbone.append(nn.BatchNorm2d(last_output_dim[0], dtype=dtype, device=device))
        backbone.append(act())
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        current_channels, current_height, current_width = last_output_dim
        # Define convolutional layers for image inputs
        for i, depth in reversed(list(enumerate(self.depths[:-1]))):
            conv = nn.ConvTranspose2d(in_channels=current_channels, out_channels=depth, kernel_size=kernel, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dtype=dtype, device=device)
            backbone.append(conv)
            backbone.append(nn.BatchNorm2d(depth, dtype=dtype, device=device))
            backbone.append(act())
            current_height, current_width = self._compute_transposed_output_dim(current_height, current_width, kernel, self.stride, self.padding, self.output_padding)
            current_channels = depth

        stride = 1 if i == 0 and first_stride else self.stride
        backbone.append(
            nn.ConvTranspose2d(
                in_channels=self.depths[0],
                out_channels=input_size[0],
                kernel_size=kernel,
                stride=stride,
                padding=self.padding,
                dtype=dtype, device=device
            )
        )
        current_height, current_width = self._compute_transposed_output_dim(current_height, current_width, kernel, stride, self.padding, 0)
        self.final_output_dim = (input_size[0], current_height, current_width)                
        self.backbone = nn.Sequential(*backbone)
        self.backbone.apply(weight_init)

    def _compute_transposed_output_dim(self, height, width, kernel_size, stride, padding, output_padding):
        new_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding
        new_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding
        return new_height, new_width
    
    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        if self._cnn_sigmoid:
            obs_hat = F.sigmoid(obs_hat)
        else:
            obs_hat += 0.5
        return obs_hat




class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, hidden_state_dim, categorical_dim, class_dim, unimix_ratio=0.01, dtype=None, device=None) -> None:
        super().__init__()
        self.stoch_dim = categorical_dim
        self.post_head = nn.Linear(image_feat_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.prior_head = nn.Linear(hidden_state_dim, categorical_dim*class_dim, dtype=dtype, device=device)
        self.unimix_ratio = unimix_ratio
        self.dtype=dtype
        self.device=device

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        if mixing_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            mixed_probs = mixing_ratio * torch.ones_like(probs, dtype=self.dtype, device=self.device) / self.stoch_dim + (1-mixing_ratio) * probs
            logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits, self.unimix_ratio)
        return logits



    

class RewardHead(nn.Module):
    def __init__(self, num_classes, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Create the backbone with dynamic number of layers
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, num_classes, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward




class TerminationHead(nn.Module):
    def __init__(self, inp_dim, hidden_units, act, layer_num, dtype=None, device=None) -> None:
        super().__init__()
        act = getattr(nn, act)

        # Create the backbone with dynamic number of layers
        layers = []
        for _ in range(layer_num):
            layers.append(nn.Linear(inp_dim, hidden_units, bias=True, dtype=dtype, device=device))
            layers.append(RMSNorm(hidden_units, dtype=dtype, device=device))
            layers.append(act())

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_units, 1, dtype=dtype, device=device)

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        distance = (obs_hat - obs)**2
        loss = reduce(distance, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, action_dim, config, device):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_state_dim = config.Models.WorldModel.HiddenStateDim
        self.final_feature_width = config.Models.WorldModel.Transformer.FinalFeatureWidth
        self.categorical_dim = config.Models.WorldModel.CategoricalDim
        self.class_dim = config.Models.WorldModel.ClassDim
        self.stoch_flattened_dim = self.categorical_dim*self.class_dim
        self.use_amp = config.BasicSettings.Use_amp
        self.use_cg = config.BasicSettings.Use_cg
        self.tensor_dtype = torch.bfloat16 if self.use_amp and not self.use_cg else config.Models.WorldModel.dtype
        self.save_every_steps = config.JointTrainAgent.SaveEverySteps
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.device = device # Maybe it's not needed
        self.model = config.Models.WorldModel.Backbone
        self.r_dim = 255
        self.mamba3_step_available = True
        self._warned_missing_mamba3_step = False
        if self.model == 'Mamba3':
            self.mamba3_step_available = getattr(mamba3_module, "mamba3_step_fn", None) is not None
        self.max_grad_norm = config.Models.WorldModel.Max_grad_norm  
        max_seq_length = max(config.JointTrainAgent.BatchLength, 
                             config.JointTrainAgent.ImagineContextLength + config.JointTrainAgent.ImagineBatchLength, 
                             config.JointTrainAgent.RealityContextLength)
        self.encoder = Encoder(
            depth=config.Models.WorldModel.Encoder.Depth,
            mults=config.Models.WorldModel.Encoder.Mults, 
            norm=config.Models.WorldModel.Encoder.Norm, 
            act=config.Models.WorldModel.Act, 
            kernel=config.Models.WorldModel.Encoder.Kernel,
            padding=config.Models.WorldModel.Encoder.Padding,
            input_size=config.Models.WorldModel.Encoder.InputSize,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        if self.model == 'Transformer':
            self.sequence_model = StochasticTransformerKVCache(
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                feat_dim=self.hidden_state_dim,
                num_layers=config.Models.WorldModel.Transformer.NumLayers,
                num_heads=config.Models.WorldModel.Transformer.NumHeads,
                max_length=max_seq_length,
                dropout=config.Models.WorldModel.Dropout
            )
        elif self.model == 'Mamba':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                r_dim=self.r_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state,
                    }
                )                                
            self.sequence_model = MambaWrapperModel(mamba_config)
        elif self.model == 'Mamba2':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                r_dim=self.r_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state, 
                    'layer': 'Mamba2'}
                )
            self.sequence_model = MambaWrapperModel(mamba_config)     
        elif self.model == 'Mamba3':
            mamba_config = MambaConfig(
                d_model=self.hidden_state_dim, 
                d_intermediate=config.Models.WorldModel.Mamba.d_intermediate,
                n_layer=config.Models.WorldModel.Mamba.n_layer,
                stoch_dim=self.stoch_flattened_dim,
                action_dim=action_dim,
                r_dim=self.r_dim,
                dropout_p=config.Models.WorldModel.Dropout,
                ssm_cfg={
                    'd_state': config.Models.WorldModel.Mamba.ssm_cfg.d_state, 
                    'layer': 'Mamba3'}
                )
            self.sequence_model = MambaWrapperModel(mamba_config)                 
        else:
            raise ValueError(f"Unknown dynamics model: {self.model}")               
        

        self.dist_head = DistHead(
            image_feat_dim=self.encoder.output_flatten_dim,
            hidden_state_dim=self.hidden_state_dim,
            categorical_dim=self.categorical_dim,
            class_dim=self.class_dim,
            unimix_ratio=config.Models.WorldModel.Unimix_ratio,
            dtype=config.Models.WorldModel.dtype, device=device
        )      
        self.image_decoder = Decoder(
            stoch_dim=self.stoch_flattened_dim,
            depth=config.Models.WorldModel.Decoder.Depth, 
            mults=config.Models.WorldModel.Decoder.Mults, 
            norm=config.Models.WorldModel.Decoder.Norm, 
            act=config.Models.WorldModel.Act, 
            kernel=config.Models.WorldModel.Decoder.Kernel, 
            padding=config.Models.WorldModel.Decoder.Padding, 
            first_stride=config.Models.WorldModel.Decoder.FirstStrideOne, 
            last_output_dim=self.encoder.output_dim,
            input_size=config.Models.WorldModel.Decoder.InputSize,
            cnn_sigmoid=config.Models.WorldModel.Decoder.FinalLayerSigmoid,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        
        self.reward_decoder = RewardHead(
            num_classes=255,
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Reward.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Reward.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.reward_decoder.apply(weight_init)
        self.termination_decoder = TerminationHead(
            inp_dim=self.hidden_state_dim,
            hidden_units=config.Models.WorldModel.Termination.HiddenUnits,
            act=config.Models.WorldModel.Act,
            layer_num=config.Models.WorldModel.Termination.LayerNum,
            dtype=config.Models.WorldModel.dtype, device=device
        )
        self.termination_decoder.apply(weight_init)

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
        self.contrastive_weight_denom = sum(self.contrastive_exp_lambda ** t for t in range(self.contrastive_steps))

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
        else:
            self.contrastive_augment = nn.Identity()
            self.contrastive_network = nn.ModuleList()
 
        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        if config.Models.WorldModel.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=config.Models.WorldModel.Laprop.LearningRate, eps=config.Models.WorldModel.Laprop.Epsilon, weight_decay=config.Models.WorldModel.Weight_decay)
        elif config.Models.WorldModel.Optimiser == 'Adam':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.Models.WorldModel.Adam.LearningRate, weight_decay=config.Models.WorldModel.Weight_decay)
        else:
            raise ValueError(f"Unknown optimiser: {config.Models.WorldModel.Optimiser}")
        # self.optimizer = AGC(self.parameters(), self.optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=config.Models.WorldModel.Warmup_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and config.Models.WorldModel.dtype is not torch.bfloat16)
    @profile
    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample
    
    # [추가] Mamba 입력용 단일 텐서 생성 헬퍼 함수
    def _prepare_mamba_input(self, latent, action, r=None):
        # action 원-핫 인코딩
        action_onehot = F.one_hot(action.long(), self.sequence_model.config.action_dim).float().to(latent.dtype)
        # r이 주어지지 않은 경우 영벡터로 채움 (예: 처음 상태)
        if r is None:
            r = torch.zeros(latent.shape[0], latent.shape[1], self.r_dim, device=latent.device, dtype=latent.dtype)
        # latent(stoch_dim) + action(action_dim) + r(255) 병합
        return torch.cat([latent, action_onehot, r], dim=-1)
    
    @profile
    # [수정] r=None 매개변수 추가
    def calc_last_dist_feat(self, latent, action, inference_params=None, r=None, is_first=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask, is_first=is_first)
            else:
                # [수정] 진짜 보상을 Two-hot 인코딩하고 시간축으로 Shift하여 r_{t-1} 형태로 Mamba에 주입
                if r is not None:
                    r_twohot = self.symlog_twohot_loss_func.encode(r).to(latent.dtype)
                    r_shifted = torch.zeros_like(r_twohot)
                    r_shifted[:, 1:, :] = r_twohot[:, :-1, :]
                else:
                    r_shifted = None
                    
                mamba_input = self._prepare_mamba_input(latent, action, r_shifted)
                dist_feat = self.sequence_model(mamba_input, inference_params=inference_params, is_first=is_first)
                
            last_dist_feat = dist_feat[:, -1:]
            prior_logits = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat
    @profile
    # 얘는 아직 _prepare_mamba_input 전처리 안 함. 나중에 쓰이면 넣자.
    def calc_last_post_feat(self, latent, action, current_obs, inference_params=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(current_obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)            
            if self.model == 'Transformer':
                temporal_mask = get_subsequent_mask(latent)
                dist_feat = self.sequence_model(latent, action, temporal_mask)
            else:
                mamba_input = self._prepare_mamba_input(latent, action, r=None) 
                dist_feat = self.sequence_model(mamba_input, inference_params=inference_params)
            last_dist_feat = dist_feat[:, -1:]
            shifted_feat = last_dist_feat
            x = torch.cat((shifted_feat, flattened_sample), -1)
            post_feat = self._obs_out_layers(x)
            post_stat = self._obs_stat_layer(post_feat)
            post_logits = post_stat.reshape(list(post_stat.shape[:-1]) + [self.categorical_dim, self.categorical_dim])
            post_sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            post_flattened_sample = self.flatten_sample(post_sample)            

        return post_flattened_sample, post_feat    
    @profile
    # only called when using Transformer
    def predict_next(self, last_flattened_sample, action, log_video=True, r=None, is_first=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if self.model == 'Transformer':
                dist_feat = self.sequence_model.forward_with_kv_cache(last_flattened_sample, action, is_first=is_first)
            else:
                # [수정] r을 함께 병합하여 전달
                mamba_input = self._prepare_mamba_input(last_flattened_sample, action, r)
                dist_feat = self.sequence_model(mamba_input, is_first=is_first)
                
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_logits = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_logits)
            
            # 다음 스텝 Mamba의 r 인자로 넘겨줄 미분 가능한 확률 벡터
            reward_probs = F.softmax(reward_logits, dim=-1)
            
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        # 반환값에 reward_probs 추가
        return obs_hat, reward_hat, reward_probs, termination_hat, prior_flattened_sample, dist_feat
    @profile
    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        # dist = Independent(
        #     OneHotDist(logits), 1
        # )        
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
            # sample = dist.sample()
        elif sample_mode == "mode":
            sample = dist.mode
            # sample = dist.mode()
        elif sample_mode == "probs":
            sample = dist.probs
        return sample
    
    
    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def _augment_for_contrastive(self, obs):
        batch_size, seq_len = obs.shape[:2]
        obs_flat = obs.flatten(start_dim=0, end_dim=1).to(torch.float32)
        obs_aug_flat = self.contrastive_augment(obs_flat)
        obs_aug = rearrange(obs_aug_flat, "(B L) C H W -> B L C H W", B=batch_size, L=seq_len)
        return obs_aug.to(dtype=obs.dtype)

    def _build_future_action_condition(self, action_onehot, horizon):
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        valid_length = action_onehot.shape[1] - horizon
        action_chunks = []
        for step in range(horizon):
            start = 1 + step
            end = start + valid_length
            action_chunks.append(action_onehot[:, start:end])

        return torch.cat(action_chunks, dim=-1)

    def _subsample_pairs(self, predictor_input, target_input):
        predictor_input = predictor_input.flatten(start_dim=0, end_dim=1)
        target_input = target_input.flatten(start_dim=0, end_dim=1)

        if self.contrastive_max_samples > 0 and predictor_input.shape[0] > self.contrastive_max_samples:
            sample_idx = torch.randperm(predictor_input.shape[0], device=predictor_input.device)[:self.contrastive_max_samples]
            predictor_input = predictor_input[sample_idx]
            target_input = target_input[sample_idx]

        return predictor_input, target_input

    def _cosine_info_nce_loss(self, predictor_output, representation_output):
        if predictor_output.dim() > 2:
            predictor_output = predictor_output.flatten(start_dim=0, end_dim=1)
        if representation_output.dim() > 2:
            representation_output = representation_output.flatten(start_dim=0, end_dim=1)

        num_samples = predictor_output.shape[0]
        if num_samples == 0:
            zero = predictor_output.new_tensor(0.0, dtype=torch.float32)
            return zero, zero

        predictor_output = F.normalize(predictor_output.float(), dim=-1)
        representation_output = F.normalize(representation_output.float(), dim=-1)

        temperature = max(self.contrastive_temperature, 1e-6)
        logits = predictor_output @ representation_output.transpose(0, 1)
        logits = logits / temperature

        targets = torch.arange(num_samples, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        acc = (logits.argmax(dim=-1) == targets).float().mean()

        return loss, acc

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype, device):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.hidden_state_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.sample_buffer = torch.zeros(latent_size, dtype=dtype, device=device)
            self.dist_feat_buffer = torch.zeros(hidden_size, dtype=dtype, device=device)
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device=device)
    @profile
    #transformer도 첫 번째 루프에서는 reward를 예측하는 게 아니라 받아오는 식으로 수정해야 함. 지금은 살짝 혼종 상태.
    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action, sample_is_first,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step):

        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        if self.model == 'Transformer':
            self.sequence_model.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        obs_hat_list = []
            
        # context
        context_latent = self.encode_obs(sample_obs)
        
        # 첫 번째 스텝을 위한 초기 r 설정 (0으로 초기화)
        last_r_encoded = torch.zeros(imagine_batch_size, 1, 255, dtype=self.tensor_dtype, device=self.device)

        # 1. 첫 번째 루프 (Context 처리)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_reward_probs, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video,
                r=last_r_encoded,
                is_first=sample_is_first[:, i:i+1]
            )
            # [수정됨] 예측된 보상 확률 분포를 다음 스텝의 r_{t-1}로 업데이트
            last_r_encoded = last_reward_probs.to(self.tensor_dtype)

        self.sample_buffer[:, 0:1] = last_latent
        self.dist_feat_buffer[:, 0:1] = last_dist_feat

        # 2. 두 번째 루프 (Imagine 생성)
        for i in range(imagine_batch_length):
            action, _ = agent.sample(torch.cat([self.sample_buffer[:, i:i+1], self.dist_feat_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_reward_probs, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.sample_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video, r=last_r_encoded)

            self.sample_buffer[:, i+1:i+2] = last_latent
            self.dist_feat_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            
            # [수정됨] 예측된 보상 확률 분포를 다음 스텝의 r_{t-1}로 업데이트
            last_r_encoded = last_reward_probs.to(self.tensor_dtype)

            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//4] * 255)  # uniform sample vec_env

        if log_video:    
            img_frames = torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 255)
            img_frames = img_frames.permute(1, 2, 3, 0, 4)
            img_frames = img_frames.reshape(imagine_batch_length, 3, 64, 64 * 4).cpu().float().detach().numpy().astype(np.uint8)
            logger.log("Imagine/predict_video", img_frames, global_step=global_step)

        return torch.cat([self.sample_buffer, self.dist_feat_buffer], dim=-1), self.action_buffer, None, None, self.reward_hat_buffer, self.termination_hat_buffer

    @profile
    def imagine_data2(self, agent: agents.ActorCriticAgent, sample_obs, sample_action, sample_reward, sample_is_first,
                     imagine_batch_size, imagine_batch_length, log_video, logger, global_step):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype, device=self.device)
        # context
        context_latent = self.encode_obs(sample_obs)
        batch_size, seqlen_og, embedding_dim = context_latent.shape
        max_length = imagine_batch_length + seqlen_og

        # Mamba3 decode step relies on optional Cute kernels. If unavailable,
        # fall back to full-sequence forward passes without inference cache.
        use_incremental_decode = not (self.model == 'Mamba3' and not self.mamba3_step_available)
        use_cg_decode = self.use_cg and use_incremental_decode
        incremental_decode_disabled_reason = None
        if self.model == 'Mamba3' and not self.mamba3_step_available:
            incremental_decode_disabled_reason = "Mamba3 step kernel is unavailable (mamba3_step_fn is None)"
        if not use_incremental_decode and not self._warned_missing_mamba3_step:
            reason = incremental_decode_disabled_reason or "unknown runtime condition"
            print(
                f"[WorldModel][Warning] Incremental decode is disabled: {reason}. "
                "Using full-sequence fallback (slower)."
            )
            self._warned_missing_mamba3_step = True
        
        # ------------------------------------------------------------------
        # [수정 1] Mamba에 실제로 들어갈 단일 텐서의 차원 계산 (latent + action + r)
        # ------------------------------------------------------------------
        action_dim = self.sequence_model.config.action_dim
        mamba_input_dim = self.stoch_flattened_dim + action_dim + 255 

        if use_cg_decode:
            if not hasattr(self.sequence_model, "_decoding_cache"):
                self.sequence_model._decoding_cache = None
            self.sequence_model._decoding_cache = update_graph_cache(
                self.sequence_model,
                self.sequence_model._decoding_cache,
                imagine_batch_size,
                seqlen_og,
                max_length,
                mamba_input_dim, # [수정 2] embedding_dim 대신 확장된 mamba_input_dim 사용
            )
            inference_params = self.sequence_model._decoding_cache.inference_params
            inference_params.reset(max_length, imagine_batch_size)
        elif use_incremental_decode:
            inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=imagine_batch_size, key_value_dtype=torch.bfloat16 if self.use_amp else None)
        else:
            inference_params = None

        # ------------------------------------------------------------------
        # [수정 3] get_hidden_state가 단일 텐서(mamba_input)만 받도록 서명 수정
        # ------------------------------------------------------------------
        def get_hidden_state(mamba_input, inference_params, is_first=None):
            if inference_params is None:
                return self.sequence_model(mamba_input, is_first=is_first)
            decoding = inference_params.seqlen_offset > 0
            has_reset = is_first is not None and bool(torch.any(is_first > 0.5))
            if not use_cg_decode or not decoding or has_reset:
                hidden_state = self.sequence_model(
                    mamba_input, 
                    inference_params=inference_params,
                    is_first=is_first,
                )
            else:
                hidden_state = self.sequence_model._decoding_cache.run(
                    mamba_input, inference_params.seqlen_offset
                )
            return hidden_state        

        def should_stop(step_idx, inference_params):
            if inference_params is None:
                return step_idx >= imagine_batch_length
            if inference_params.seqlen_offset == 0:
                return False
            if inference_params.seqlen_offset >= max_length:
                return True
            return False

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp and not use_cg_decode):
            # 1. 실제 보상을 255차원 Two-hot으로 변환
            r_twohot = self.symlog_twohot_loss_func.encode(sample_reward).to(self.tensor_dtype)
            
            # 2. t-1 시점을 참조하기 위해 텐서를 우측으로 한 칸 Shift
            context_r = torch.zeros_like(r_twohot)
            context_r[:, 1:, :] = r_twohot[:, :-1, :]
            
            # ------------------------------------------------------------------
            # [수정 4] Context 병렬 처리 시 단일 mamba_input 생성 후 전달
            # ------------------------------------------------------------------
            context_mamba_input = self._prepare_mamba_input(context_latent, sample_action, context_r)
            context_dist_feat = get_hidden_state(context_mamba_input, inference_params, is_first=sample_is_first)
            
            if inference_params is not None:
                inference_params.seqlen_offset += context_dist_feat.shape[1]
            context_prior_logits = self.dist_head.forward_prior(context_dist_feat)
            context_prior_sample = self.stright_throught_gradient(context_prior_logits)
            context_flattened_sample = self.flatten_sample(context_prior_sample)

            running_mamba_input = context_mamba_input if inference_params is None else None
            running_is_first = sample_is_first if inference_params is None else None

            dist_feat_list, sample_list  = [context_dist_feat[:, -1:]], [context_flattened_sample[:, -1:]]
            self.sample_buffer[:, 0:1] = context_flattened_sample[:, -1:]
            self.dist_feat_buffer[:, 0:1] = context_dist_feat[:, -1:]
            action_list, old_logits_list = [], []
            i = 0
            
            # 상상(Imagine) 루프 시작 전, 진짜 보상 궤적을 토대로 형성된 
            # 마지막 Context 스텝의 은닉 상태로 보상을 예측하여 넘겨줌
            context_last_reward_logits = self.reward_decoder(context_dist_feat[:, -1:])
            last_r_encoded = F.softmax(context_last_reward_logits, dim=-1).to(self.tensor_dtype)

            # 자가회귀(Autoregressive) 상상 루프
            while not should_stop(i, inference_params):
                action, logits = agent.sample(torch.cat([self.sample_buffer[:, i:i+1], self.dist_feat_buffer[:, i:i+1]], dim=-1))
                action_list.append(action)
                self.action_buffer[:, i:i+1] = action
                old_logits_list.append(logits)
                
                # ------------------------------------------------------------------
                # [수정 5] Imagine 스텝 루프 내에서도 단일 mamba_input 생성 후 전달
                # ------------------------------------------------------------------
                step_mamba_input = self._prepare_mamba_input(sample_list[-1], action_list[-1], last_r_encoded)
                if inference_params is None:
                    running_mamba_input = torch.cat([running_mamba_input, step_mamba_input], dim=1)
                    running_is_first = torch.cat(
                        [running_is_first, torch.zeros((running_is_first.shape[0], 1), device=running_is_first.device, dtype=running_is_first.dtype)],
                        dim=1,
                    )
                    dist_feat = get_hidden_state(running_mamba_input, inference_params, is_first=running_is_first)[:, -1:]
                else:
                    dist_feat = get_hidden_state(step_mamba_input, inference_params)
                
                dist_feat_list.append(dist_feat)
                self.dist_feat_buffer[:, i+1:i+2] = dist_feat
                if inference_params is not None:
                    inference_params.seqlen_offset += sample_list[-1].shape[1]

                current_reward_logits = self.reward_decoder(dist_feat)
                last_r_encoded = F.softmax(current_reward_logits, dim=-1).to(self.tensor_dtype)

                prior_logits = self.dist_head.forward_prior(dist_feat_list[-1])
                prior_sample = self.stright_throught_gradient(prior_logits)
                prior_flattened_sample = self.flatten_sample(prior_sample)
                sample_list.append(prior_flattened_sample)
                self.sample_buffer[:, i+1:i+2] = prior_flattened_sample
                i += 1
                    
                        
            # sample_tensor = torch.cat(sample_list, dim=1)
            # dist_feat_tensor = torch.cat(dist_feat_list, dim=1)
            # action_tensor = torch.cat(action_list, dim=1)
            old_logits_tensor = torch.cat(old_logits_list, dim=1)

            reward_hat_tensor = self.reward_decoder(self.dist_feat_buffer[:,:-1])
            self.reward_hat_buffer = self.symlog_twohot_loss_func.decode(reward_hat_tensor)
            termination_hat_tensor = self.termination_decoder(self.dist_feat_buffer[:,:-1])
            self.termination_hat_buffer = termination_hat_tensor > 0


            if log_video:
                obs_hat = self.image_decoder(self.sample_buffer[::imagine_batch_size//4]) * 255
                obs_hat = torch.clamp(obs_hat, 0, 255)
                img_frames = obs_hat.permute(1, 2, 3, 0, 4)
                img_frames = img_frames.reshape(imagine_batch_length+1, 3, 64, 64 * 4).cpu().float().detach().numpy().astype(np.uint8)
                logger.log("Imagine/predict_video", img_frames, global_step=global_step)
        return torch.cat([self.sample_buffer, self.dist_feat_buffer], dim=-1), self.action_buffer, old_logits_tensor, torch.cat([context_flattened_sample, context_dist_feat], dim=-1), self.reward_hat_buffer, self.termination_hat_buffer


    @profile
    def update(self, obs, action, reward, termination, is_first, global_step, epoch_step, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)

            # ---------------------------------------------------------
            # [NEW] Selection Mechanism을 위한 Reward 인코딩 및 Shift 적용
            # ---------------------------------------------------------
            r_twohot = self.symlog_twohot_loss_func.encode(reward).to(flattened_sample.dtype)
            r_shifted = torch.zeros_like(r_twohot)
            r_shifted[:, 1:, :] = r_twohot[:, :-1, :]
            is_first = is_first.to(device=flattened_sample.device, dtype=flattened_sample.dtype)
            # ---------------------------------------------------------

            # dynamics models
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

                valid_steps = 0
                for t in range(self.contrastive_steps):
                    if batch_length - t <= 0:
                        break

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

                    step_weight = (self.contrastive_exp_lambda ** t) / self.contrastive_weight_denom
                    contrastive_loss = contrastive_loss + self.contrastive_loss_scale * step_weight * step_loss.to(contrastive_loss.dtype)
                    contrastive_acc = contrastive_acc + step_acc.to(contrastive_acc.dtype)
                    valid_steps += 1

                if valid_steps > 0:
                    contrastive_acc = contrastive_acc / valid_steps

            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat[:batch_size], obs[:batch_size])
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            total_loss = reconstruction_loss + reward_loss + termination_loss + dynamics_loss + 0.1*representation_loss + contrastive_loss

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.warmup_scheduler.dampen()

        if (global_step + epoch_step) % self.save_every_steps == 0: # and global_step != 0:
            sample_obs = torch.clamp(obs[:3, 0, :]*255, 0, 255).permute(0, 2, 3, 1).cpu().detach().float().numpy().astype(np.uint8)
            sample_obs_hat = torch.clamp(obs_hat[:3, 0, :]*255, 0, 255).permute(0, 2, 3, 1).cpu().detach().float().numpy().astype(np.uint8)

            concatenated_images = []
            for idx in range(3):
                concatenated_image = np.concatenate((sample_obs[idx], sample_obs_hat[idx]), axis=0)  # Concatenate vertically
                concatenated_images.append(concatenated_image)

            # Combine selected images into one image
            final_image = np.concatenate(concatenated_images, axis=1)  # Concatenate horizontally
            height, width, _ = final_image.shape
            scale_factor = 6
            final_image_resized = cv2.resize(final_image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_NEAREST)
            if logger is not None:
                logger.log("Reconstruct/Reconstructed images", [final_image_resized], global_step=global_step)

        return  reconstruction_loss.item(), reward_loss.item(), termination_loss.item(), \
                dynamics_loss.item(), dynamics_real_kl_div.item(), representation_loss.item(), \
                representation_real_kl_div.item(), total_loss.item(), contrastive_loss.item(), contrastive_acc.item()
