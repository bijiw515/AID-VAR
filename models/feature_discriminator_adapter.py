"""
 -

VQVAE
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class FeatureDiscriminator(nn.Module):
    """

    
 VQVAE f_BChw
 : (B, z_channels=32, H//8, W//8) for 256x256 input
    """
    
    def __init__(
        self, 
        in_channels: int = 32,  # VQVAE z_channels
        base_channels: int = 64,
        num_layers: int = 4,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
 #
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            next_channels = base_channels * (2 ** i)
            stride = 2 if i > 0 else 1  #
            
            conv = nn.Conv2d(
                current_channels, 
                next_channels, 
                kernel_size=4, 
                stride=stride, 
                padding=1,
                bias=False
            )
            
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.extend([
                conv,
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            current_channels = next_channels
        
 #
        self.feature_extractor = nn.Sequential(*layers)
        
 # +
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        classifier = nn.Linear(current_channels, 1)
        if use_spectral_norm:
            classifier = nn.utils.spectral_norm(classifier)
        
        self.classifier = classifier
        
 # StyleGAN-T
        self.multi_scale_heads = nn.ModuleDict()
        for scale in [4, 8, 16]:  #
            head = nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Flatten(),
                nn.utils.spectral_norm(nn.Linear(current_channels * scale * scale, 1)) if use_spectral_norm
                else nn.Linear(current_channels * scale * scale, 1)
            )
            self.multi_scale_heads[f'scale_{scale}'] = head
        
 logger.info(f"️ :")
 logger.info(f" : {in_channels}")
 logger.info(f" : {base_channels}")
 logger.info(f" : {num_layers}")
 logger.info(f" : {use_spectral_norm}")
 logger.info(f" : {list(self.multi_scale_heads.keys())}")
    
    def forward(self, features: torch.Tensor, multi_scale: bool = True) -> torch.Tensor:
        """

        
        Args:
 features: VQVAE (B, z_channels, H//8, W//8)
 multi_scale:
            
        Returns:
 logits: logits (B, n_heads) (B, 1)
        """
 #
        if features.dim() != 4:
 raise ValueError(f"4D{features.dim()}D: {features.shape}")
        
        if features.size(1) != self.in_channels:
 raise ValueError(f"{self.in_channels}{features.size(1)}")
        
 #
        if torch.isnan(features).any() or torch.isinf(features).any():
 logger.warning("️ NaN/Inf")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
 #
        features = torch.clamp(features, min=-10.0, max=10.0)
        
 #
        h = self.feature_extractor(features)
        
        if multi_scale:
 #
            logits_list = []
            
 #
            pooled = self.global_pool(h).flatten(1)
            main_logits = self.classifier(pooled)
            logits_list.append(main_logits)
            
 #
            for scale_name, head in self.multi_scale_heads.items():
                try:
                    scale_logits = head(h)
                    logits_list.append(scale_logits)
                except Exception as e:
 logger.warning(f"️ {scale_name}: {e}")
 # fallback
                    fallback = torch.zeros_like(main_logits)
                    logits_list.append(fallback)
            
 # : (B, n_heads)
            logits = torch.cat(logits_list, dim=1)
            
        else:
 #
            pooled = self.global_pool(h).flatten(1)
            logits = self.classifier(pooled)
        
        return logits


class FeatureDiscriminatorAdapter(nn.Module):
    """
 -
    

 - 🆕 VQVAE encoder (f_BChw)
 -
 - RGB
 - StyleGAN
    """
    
    def __init__(
        self, 
        vae,  # VQ-VAE
        feature_channels: int = 32,  # VQVAE z_channels
        base_channels: int = 64,
        num_layers: int = 4,
        img_resolution: int = 256,
        c_dim: int = 0,  #
        use_spectral_norm: bool = True,
        freeze_vae: bool = True
    ):
        super().__init__()
        
        self.vae = vae
        self.feature_channels = feature_channels
        self.img_resolution = img_resolution
        self.c_dim = c_dim
        
 # VQ-VAE
        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
 logger.info(" VQ-VAE")
        
 #
        self.discriminator = FeatureDiscriminator(
            in_channels=feature_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            use_spectral_norm=use_spectral_norm
        )
        
 #
        self.feature_spatial_size = img_resolution // self.vae.downsample
        
 logger.info(f" ...")
 logger.info(f" : {img_resolution}")
 logger.info(f" : {self.feature_spatial_size}x{self.feature_spatial_size}")
 logger.info(f" : {feature_channels}")
 logger.info(f" VQ-VAE: {self.vae.downsample}")
        
 #
        self._log_parameter_statistics()
    
    def _log_parameter_statistics(self):
 """"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_params())
        frozen_params = total_params - trainable_params
        
 logger.info(" :")
 logger.info(f" : {total_params:,}")
 logger.info(f" (VQ-VAE): {frozen_params:,}")
 logger.info(f" (): {trainable_params:,}")
 logger.info(f" : {trainable_params/total_params*100:.2f}%")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
 """"""
        trainable_params = []
        for param in self.discriminator.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def train(self, mode: bool = True):
 """trainVQ-VAEeval"""
        super().train(mode)
        if hasattr(self, 'vae'):
            self.vae.eval()  # VQ-VAEeval
        return self
    
    def forward_features(self, features: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
 🆕
        
        Args:
 features: VQVAE (B, z_channels, H//8, W//8)
 c:
            
        Returns:
 logits: logits (B, n_heads)
        """
        return self.discriminator(features, multi_scale=True)
    
    def forward_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
 RGB
        
 StyleGAN

        
        Args:
 rgb_images: RGB (B, 3, H, W) [-1, 1]
 c:
            
        Returns:
 logits: logits (B, n_heads)
        """
 # RGB
        with torch.no_grad():  # VQ-VAE
            features = self.vae.get_encoder_features(rgb_images)
        
 #
        return self.forward_features(features, c)
    
    def forward_soft_rgb(self, rgb_images: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
 -
        
 forward_rgb
 →
        """
        return self.forward_rgb(rgb_images, c)
    
    def forward(self, 
                inputs: torch.Tensor,  # RGB
                mode: str = "auto",     # "rgb", "features", "auto"
                c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        
        Args:
 inputs: RGB (B, 3, H, W) (B, z_channels, H//8, W//8)
 mode: - "rgb", "features", "auto"
 c:
        """
        if mode == "auto":
 #
            if inputs.size(1) == 3:  # RGB
                return self.forward_rgb(inputs, c)
            elif inputs.size(1) == self.feature_channels:  #
                return self.forward_features(inputs, c)
            else:
 raise ValueError(f": {inputs.size(1)}")
        
        elif mode == "rgb":
            return self.forward_rgb(inputs, c)
        elif mode == "features":
            return self.forward_features(inputs, c)
        else:
 raise ValueError(f": {mode}")


#
# FeatureDiscriminator FeatureDiscriminatorAdapter