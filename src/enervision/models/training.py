"""
Training and Fine-tuning Module for IBM Granite TTM Models
Implements efficient few-shot learning for Indian buildings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import yaml
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BuildingEnergyDataset(Dataset):
    """Dataset for Indian building energy consumption"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 context_length: int = 512,
                 prediction_length: int = 96,
                 stride: int = 24,
                 features: List[str] = None):
        
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        
        if features is None:
            self.features = ['energy_consumption', 'temperature', 'humidity', 
                           'solar_irradiance', 'occupancy_rate']
        else:
            self.features = features
        
        # Create samples
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Dict]:
        """Create training samples with sliding window"""
        samples = []
        
        total_length = self.context_length + self.prediction_length
        
        for i in range(0, len(self.data) - total_length + 1, self.stride):
            # Context window
            context_data = self.data.iloc[i:i+self.context_length]
            
            # Prediction window
            target_data = self.data.iloc[i+self.context_length:i+total_length]
            
            # Extract features
            context_features = context_data[self.features].values
            target_values = target_data['energy_consumption'].values
            
            # Time features
            time_features = self._extract_time_features(context_data)
            
            sample = {
                'context': context_features,
                'target': target_values,
                'time_features': time_features,
                'metadata': {
                    'start_time': context_data.index[0],
                    'end_time': target_data.index[-1]
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def _extract_time_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract temporal features"""
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
        else:
            timestamps = data.index
        
        time_features = np.column_stack([
            timestamps.hour,
            timestamps.dayofweek,
            timestamps.day,
            timestamps.month
        ])
        
        return time_features
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        return {
            'context': torch.FloatTensor(sample['context']),
            'target': torch.FloatTensor(sample['target']),
            'time_features': torch.LongTensor(sample['time_features'])
        }


class TTMTrainer:
    """Trainer for IBM Granite TTM models with Indian context"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss function
        self.criterion = self._setup_loss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw')
        
        if opt_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001),
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=opt_config.get('betas', [0.9, 0.999])
            )
        elif opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 0.001),
                momentum=0.9
            )
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine')
        
        if sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 100),
                eta_min=sched_config.get('min_lr', 1e-5)
            )
        elif sched_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        else:
            return None
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function"""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'mse')
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        else:
            return nn.MSELoss()
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move to device
            context = batch['context'].to(self.device)
            target = batch['target'].to(self.device)
            time_features = batch.get('time_features')
            
            if time_features is not None:
                time_features = time_features.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(context, time_features)
            
            # Calculate loss
            loss = self.criterion(predictions, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)
                time_features = batch.get('time_features')
                
                if time_features is not None:
                    time_features = time_features.to(self.device)
                
                # Forward pass
                predictions = self.model(context, time_features)
                
                # Calculate loss
                loss = self.criterion(predictions, target)
                
                # Track loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, 
             train_loader: DataLoader, 
             val_loader: Optional[DataLoader] = None,
             num_epochs: Optional[int] = None) -> Dict:
        """Full training loop"""
        
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 100)
        
        early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pth')
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                print(f"Train Loss: {train_loss:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.history['learning_rates'].append(current_lr)
                print(f"Learning Rate: {current_lr:.6f}")
        
        logger.info("Training completed")
        
        return self.history
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': len(self.history['train_loss']),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {filepath}")


class TTMFineTuner:
    """Fine-tuning module for Indian building adaptation"""
    
    def __init__(self, base_model: nn.Module, config: Dict):
        self.base_model = base_model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def few_shot_fine_tune(self, 
                           train_data: pd.DataFrame,
                           val_data: Optional[pd.DataFrame] = None,
                           sample_ratio: float = 0.05) -> nn.Module:
        """Fine-tune with only 5% of data (TTM efficiency)"""
        
        logger.info(f"Few-shot fine-tuning with {sample_ratio*100}% of data")
        
        # Sample data
        n_samples = int(len(train_data) * sample_ratio)
        train_subset = train_data.sample(n=n_samples, random_state=42)
        
        # Create datasets
        train_dataset = BuildingEnergyDataset(
            train_subset,
            context_length=self.config.get('context_length', 512),
            prediction_length=self.config.get('prediction_length', 96)
        )
        
        val_dataset = None
        if val_data is not None:
            val_subset = val_data.sample(n=int(len(val_data) * sample_ratio), random_state=42)
            val_dataset = BuildingEnergyDataset(
                val_subset,
                context_length=self.config.get('context_length', 512),
                prediction_length=self.config.get('prediction_length', 96)
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=2
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=2
            )
        
        # Setup trainer with reduced epochs for few-shot
        fine_tune_config = self.config.copy()
        fine_tune_config['num_epochs'] = 20  # Reduced epochs for few-shot
        fine_tune_config['learning_rate'] = 0.0001  # Lower learning rate
        
        trainer = TTMTrainer(self.base_model, fine_tune_config, self.device)
        
        # Fine-tune
        history = trainer.train(train_loader, val_loader)
        
        logger.info(f"Fine-tuning completed. Best val loss: {trainer.best_val_loss:.4f}")
        
        return self.base_model
    
    def adapt_to_indian_context(self, 
                                model: nn.Module,
                                indian_features: Dict) -> nn.Module:
        """Add Indian context adaptation layers"""
        
        # Create adapter module
        class IndianAdapter(nn.Module):
            def __init__(self, base_model, adapter_dim=16):
                super().__init__()
                self.base_model = base_model
                
                # Indian context encoders
                self.climate_encoder = nn.Embedding(4, adapter_dim)  # 4 climate zones
                self.building_encoder = nn.Embedding(6, adapter_dim)  # 6 building types
                self.festival_encoder = nn.Linear(8, adapter_dim)  # 8 festivals
                self.monsoon_encoder = nn.Linear(3, adapter_dim)  # Monsoon features
                
                # Fusion layer
                self.fusion = nn.Sequential(
                    nn.Linear(adapter_dim * 4, adapter_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(adapter_dim * 2, adapter_dim),
                    nn.Sigmoid()
                )
            
            def forward(self, x, indian_context):
                # Base model output
                base_output = self.base_model(x)
                
                # Encode Indian features
                climate = self.climate_encoder(indian_context['climate_zone'])
                building = self.building_encoder(indian_context['building_type'])
                festival = self.festival_encoder(indian_context['festival_features'])
                monsoon = self.monsoon_encoder(indian_context['monsoon_features'])
                
                # Concatenate and fuse
                context_features = torch.cat([climate, building, festival, monsoon], dim=-1)
                adaptation_gate = self.fusion(context_features)
                
                # Apply gated adaptation
                adapted_output = base_output * (1 + adaptation_gate.unsqueeze(-1))
                
                return adapted_output
        
        # Wrap model with adapter
        adapted_model = IndianAdapter(model)
        
        logger.info("Indian context adaptation layers added")
        
        return adapted_model


class ModelCheckpointer:
    """Model checkpointing and versioning"""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_model(self, 
                  model: nn.Module,
                  metadata: Dict,
                  version: Optional[str] = None) -> str:
        """Save model with versioning"""
        
        if version is None:
            from datetime import datetime
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        checkpoint_path = self.checkpoint_dir / f'ttm_model_{version}.pth'
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata,
            'version': version
        }, checkpoint_path)
        
        # Save metadata separately
        metadata_path = self.checkpoint_dir / f'metadata_{version}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_model(self, 
                  model_class: type,
                  version: str,
                  **model_kwargs) -> Tuple[nn.Module, Dict]:
        """Load specific model version"""
        
        checkpoint_path = self.checkpoint_dir / f'ttm_model_{version}.pth'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model version {version} not found")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model instance
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded: {checkpoint_path}")
        
        return model, checkpoint['metadata']
    
    def list_versions(self) -> List[str]:
        """List available model versions"""
        versions = []
        
        for checkpoint_path in self.checkpoint_dir.glob('ttm_model_*.pth'):
            version = checkpoint_path.stem.replace('ttm_model_', '')
            versions.append(version)
        
        return sorted(versions, reverse=True)


# Export classes
__all__ = [
    'BuildingEnergyDataset',
    'TTMTrainer',
    'TTMFineTuner',
    'ModelCheckpointer'
]