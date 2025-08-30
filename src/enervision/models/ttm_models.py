"""
IBM Granite TinyTimeMixers (TTM) Model Implementations
Specialized models for energy forecasting with <1M parameters
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class RevIN(nn.Module):
    """Reversible Instance Normalization for better time series forecasting"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x
    
    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(2, x.ndim))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / self.affine_weight
        x = x * self.stdev
        x = x + self.mean
        return x


class TemporalEmbedding(nn.Module):
    """Temporal embeddings for encoding time-related features"""
    
    def __init__(self, d_model: int, embed_type: str = 'timeF'):
        super().__init__()
        
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        
        if embed_type == 'timeF':
            self.hour_embed = nn.Embedding(hour_size, d_model)
            self.weekday_embed = nn.Embedding(weekday_size, d_model)
            self.day_embed = nn.Embedding(day_size, d_model)
            self.month_embed = nn.Embedding(month_size, d_model)
        else:
            self.embed = nn.Linear(4, d_model)
    
    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        # x_mark: [batch, seq_len, 4] - hour, day, weekday, month
        if hasattr(self, 'hour_embed'):
            hour_x = self.hour_embed(x_mark[..., 0].long())
            weekday_x = self.weekday_embed(x_mark[..., 1].long())
            day_x = self.day_embed(x_mark[..., 2].long())
            month_x = self.month_embed(x_mark[..., 3].long())
            
            return hour_x + weekday_x + day_x + month_x
        else:
            return self.embed(x_mark)


class GraniteTimeMixer(nn.Module):
    """Core Granite Time Mixer block for temporal pattern learning"""
    
    def __init__(self, 
                 context_length: int,
                 prediction_length: int,
                 n_channels: int,
                 d_model: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_channels = n_channels
        self.d_model = d_model
        
        # Time mixing layers (across time dimension)
        self.time_mixing = nn.Sequential(
            nn.Linear(context_length, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )
        
        # Channel mixing layers (across channel dimension)
        self.channel_mixing = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_channels),
            nn.GELU()
        )
        
        # Cross-dimensional attention
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, prediction_length)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, 
                time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch, channels, context_length]
        time_features: [batch, context_length, d_time]
        """
        batch_size, n_channels, seq_len = x.shape
        
        # Time mixing
        time_mixed = self.time_mixing(x)  # [batch, channels, d_model]
        time_mixed = self.norm1(time_mixed)
        
        # Channel mixing with residual
        x_transposed = x.transpose(1, 2)  # [batch, context, channels]
        channel_mixed = self.channel_mixing(x_transposed)
        channel_mixed = channel_mixed.transpose(1, 2)  # [batch, channels, context]
        
        # Cross-dimensional attention
        if time_features is not None:
            # Incorporate temporal features
            attn_out, _ = self.cross_attention(time_mixed, time_features, time_features)
            time_mixed = time_mixed + attn_out
            time_mixed = self.norm2(time_mixed)
        
        # Generate predictions for each channel
        predictions = self.prediction_head(time_mixed)  # [batch, channels, pred_length]
        
        return predictions


class TTMFineTuner(nn.Module):
    """Fine-tuning adapter for Indian building context"""
    
    def __init__(self, 
                 base_model: nn.Module,
                 n_indian_features: int = 10,
                 adapter_dim: int = 16):
        super().__init__()
        
        self.base_model = base_model
        
        # Adapter layers for Indian context
        self.indian_adapter = nn.Sequential(
            nn.Linear(n_indian_features, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, adapter_dim),
            nn.ReLU()
        )
        
        # Feature fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(adapter_dim * 2, adapter_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(adapter_dim, 1)
    
    def forward(self, x: torch.Tensor, 
                indian_features: torch.Tensor) -> torch.Tensor:
        """
        x: Standard input features
        indian_features: Indian context features (climate, building type, etc.)
        """
        # Base model predictions
        base_output = self.base_model(x)
        
        # Process Indian features
        indian_context = self.indian_adapter(indian_features)
        
        # Adaptive fusion based on context
        combined = torch.cat([base_output.mean(dim=-1), indian_context], dim=-1)
        gate = self.fusion_gate(combined)
        
        # Apply gated fusion
        adapted_output = base_output * gate.unsqueeze(-1) + \
                        indian_context.unsqueeze(-1) * (1 - gate.unsqueeze(-1))
        
        return adapted_output


class UltraLightTTM(nn.Module):
    """Ultra-lightweight TTM variant (<1M parameters) for edge deployment"""
    
    def __init__(self,
                 context_length: int = 96,
                 prediction_length: int = 24,
                 n_channels: int = 7,
                 d_model: int = 32,
                 n_blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.n_channels = n_channels
        
        # Reversible Instance Normalization
        self.revin = RevIN(n_channels)
        
        # Lightweight encoder
        self.encoder = nn.ModuleList([
            self._make_encoder_block(context_length if i == 0 else d_model, 
                                    d_model, dropout)
            for i in range(n_blocks)
        ])
        
        # Temporal embedding
        self.temporal_embed = TemporalEmbedding(d_model)
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Linear(d_model * n_blocks, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, prediction_length * n_channels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_encoder_block(self, input_dim: int, output_dim: int, 
                           dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch, seq_len, channels]
        x_mark: [batch, seq_len, 4] - temporal features
        """
        batch_size, seq_len, n_channels = x.shape
        
        # Normalize
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        x = self.revin(x, mode='norm')
        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        
        # Add temporal embeddings if available
        if x_mark is not None:
            temporal_emb = self.temporal_embed(x_mark)
            x = x + temporal_emb.unsqueeze(-1).expand(-1, -1, n_channels)
        
        # Encode with skip connections
        encoder_outputs = []
        h = x.reshape(batch_size, -1)  # Flatten for first layer
        
        for encoder_block in self.encoder:
            h = encoder_block(h)
            encoder_outputs.append(h)
        
        # Concatenate all encoder outputs (skip connections)
        h = torch.cat(encoder_outputs, dim=-1)
        
        # Decode
        output = self.decoder(h)
        output = output.reshape(batch_size, self.prediction_length, n_channels)
        
        # Denormalize
        output = output.transpose(1, 2)  # [batch, channels, pred_len]
        output = self.revin(output, mode='denorm')
        output = output.transpose(1, 2)  # [batch, pred_len, channels]
        
        return output


class AnomalyDetectorTTM(nn.Module):
    """TTM-based anomaly detection model"""
    
    def __init__(self,
                 input_dim: int = 7,
                 hidden_dim: int = 32,
                 latent_dim: int = 8):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns reconstruction and latent representation"""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores based on reconstruction error"""
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction='none')
        anomaly_scores = torch.mean(mse, dim=-1)
        return anomaly_scores


class EnsembleTTM(nn.Module):
    """Ensemble of multiple TTM models for robust predictions"""
    
    def __init__(self, model_configs: List[Dict]):
        super().__init__()
        
        self.models = nn.ModuleList()
        self.weights = []
        
        for config in model_configs:
            if config['type'] == 'standard':
                model = GraniteTimeMixer(**config['params'])
            elif config['type'] == 'ultralight':
                model = UltraLightTTM(**config['params'])
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
            
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate ensemble predictions"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model(x, **kwargs)
            predictions.append(pred * weight)
        
        # Weighted average
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        
        return ensemble_pred
    
    def get_uncertainty(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prediction uncertainty from ensemble"""
        predictions = []
        
        for model in self.models:
            pred = model(x, **kwargs)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Mean and standard deviation
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


def create_ttm_model(model_type: str = 'standard', **kwargs) -> nn.Module:
    """Factory function to create TTM models"""
    
    if model_type == 'standard':
        return GraniteTimeMixer(**kwargs)
    elif model_type == 'ultralight':
        return UltraLightTTM(**kwargs)
    elif model_type == 'anomaly':
        return AnomalyDetectorTTM(**kwargs)
    elif model_type == 'ensemble':
        return EnsembleTTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Model configurations for different context lengths
TTM_CONFIGS = {
    'ttm-512-96': {
        'context_length': 512,
        'prediction_length': 96,
        'n_channels': 7,
        'd_model': 64,
        'n_heads': 4,
        'dropout': 0.1
    },
    'ttm-1024-96': {
        'context_length': 1024,
        'prediction_length': 96,
        'n_channels': 7,
        'd_model': 64,
        'n_heads': 4,
        'dropout': 0.1
    },
    'ttm-1536-96': {
        'context_length': 1536,
        'prediction_length': 96,
        'n_channels': 7,
        'd_model': 64,
        'n_heads': 4,
        'dropout': 0.1
    },
    'ttm-ultralight': {
        'context_length': 96,
        'prediction_length': 24,
        'n_channels': 7,
        'd_model': 32,
        'n_blocks': 2,
        'dropout': 0.1
    }
}