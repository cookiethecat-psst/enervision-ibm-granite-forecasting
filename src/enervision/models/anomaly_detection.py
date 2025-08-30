"""
Anomaly Detection Module using IBM Time Series Pulse approach
Real-time anomaly detection for energy consumption patterns
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Container for anomaly detection results"""
    scores: np.ndarray
    anomalies: np.ndarray
    severity: np.ndarray
    explanations: List[str]
    statistics: Dict[str, float]


class TSPulseDetector:
    """IBM Time Series Pulse inspired anomaly detector"""
    
    def __init__(self, 
                 window_size: int = 24,
                 threshold_sigma: float = 3.0,
                 min_consecutive: int = 3):
        
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        self.min_consecutive = min_consecutive
        self.scaler = StandardScaler()
        
        # Historical statistics
        self.historical_mean = None
        self.historical_std = None
        self.seasonal_patterns = {}
        
    def fit(self, historical_data: np.ndarray, 
            timestamps: Optional[pd.DatetimeIndex] = None):
        """Fit detector on historical data"""
        
        # Calculate baseline statistics
        self.historical_mean = np.mean(historical_data)
        self.historical_std = np.std(historical_data)
        
        # Fit scaler
        self.scaler.fit(historical_data.reshape(-1, 1))
        
        # Learn seasonal patterns if timestamps provided
        if timestamps is not None:
            self._learn_seasonal_patterns(historical_data, timestamps)
        
        logger.info(f"TSPulse detector fitted on {len(historical_data)} samples")
        
    def _learn_seasonal_patterns(self, data: np.ndarray, 
                                timestamps: pd.DatetimeIndex):
        """Learn hourly, daily, and weekly patterns"""
        
        df = pd.DataFrame({'value': data, 'timestamp': timestamps})
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        # Hourly patterns
        self.seasonal_patterns['hourly'] = df.groupby('hour')['value'].agg(['mean', 'std']).to_dict()
        
        # Day of week patterns
        self.seasonal_patterns['weekly'] = df.groupby('dayofweek')['value'].agg(['mean', 'std']).to_dict()
        
    def detect(self, data: np.ndarray, 
              timestamps: Optional[pd.DatetimeIndex] = None) -> AnomalyResult:
        """Detect anomalies in time series data"""
        
        # Calculate multiple anomaly scores
        scores = self._calculate_composite_scores(data, timestamps)
        
        # Determine anomalies
        anomalies = scores > self.threshold_sigma
        
        # Apply consecutive filter
        anomalies = self._filter_consecutive(anomalies)
        
        # Calculate severity
        severity = self._calculate_severity(scores)
        
        # Generate explanations
        explanations = self._generate_explanations(data, scores, anomalies)
        
        # Calculate statistics
        statistics = {
            'total_anomalies': np.sum(anomalies),
            'anomaly_rate': np.mean(anomalies) * 100,
            'max_severity': np.max(severity),
            'mean_score': np.mean(scores)
        }
        
        return AnomalyResult(
            scores=scores,
            anomalies=anomalies,
            severity=severity,
            explanations=explanations,
            statistics=statistics
        )
    
    def _calculate_composite_scores(self, data: np.ndarray, 
                                   timestamps: Optional[pd.DatetimeIndex]) -> np.ndarray:
        """Calculate composite anomaly scores"""
        
        scores = []
        
        # Statistical anomaly score
        stat_scores = self._statistical_anomaly_score(data)
        scores.append(stat_scores)
        
        # Isolation score
        iso_scores = self._isolation_score(data)
        scores.append(iso_scores)
        
        # Seasonal deviation score
        if timestamps is not None and self.seasonal_patterns:
            seasonal_scores = self._seasonal_deviation_score(data, timestamps)
            scores.append(seasonal_scores)
        
        # Combine scores
        composite_scores = np.mean(scores, axis=0)
        
        return composite_scores
    
    def _statistical_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Calculate statistical anomaly scores"""
        
        # Rolling statistics
        scores = np.zeros(len(data))
        
        for i in range(len(data)):
            start_idx = max(0, i - self.window_size)
            window = data[start_idx:i+1]
            
            if len(window) > 1:
                mean = np.mean(window)
                std = np.std(window)
                
                if std > 0:
                    scores[i] = abs((data[i] - mean) / std)
                else:
                    scores[i] = 0
            else:
                scores[i] = 0
        
        return scores
    
    def _isolation_score(self, data: np.ndarray) -> np.ndarray:
        """Calculate isolation-based anomaly scores"""
        
        # Simplified isolation score
        scores = np.zeros(len(data))
        
        for i in range(len(data)):
            # Compare with recent history
            start_idx = max(0, i - self.window_size)
            window = data[start_idx:i+1]
            
            if len(window) > 1:
                # Count how many values are more extreme
                isolation = np.sum(np.abs(window - data[i]) > np.std(window))
                scores[i] = isolation / len(window)
            else:
                scores[i] = 0
        
        return scores
    
    def _seasonal_deviation_score(self, data: np.ndarray, 
                                 timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Calculate deviation from seasonal patterns"""
        
        scores = np.zeros(len(data))
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            dow = ts.dayofweek
            
            # Expected value based on patterns
            hourly_mean = self.seasonal_patterns['hourly']['mean'].get(hour, self.historical_mean)
            hourly_std = self.seasonal_patterns['hourly']['std'].get(hour, self.historical_std)
            
            weekly_mean = self.seasonal_patterns['weekly']['mean'].get(dow, self.historical_mean)
            weekly_std = self.seasonal_patterns['weekly']['std'].get(dow, self.historical_std)
            
            # Combined expectation
            expected = (hourly_mean + weekly_mean) / 2
            expected_std = np.sqrt((hourly_std**2 + weekly_std**2) / 2)
            
            if expected_std > 0:
                scores[i] = abs((data[i] - expected) / expected_std)
            else:
                scores[i] = 0
        
        return scores
    
    def _filter_consecutive(self, anomalies: np.ndarray) -> np.ndarray:
        """Filter anomalies based on consecutive occurrence"""
        
        filtered = anomalies.copy()
        consecutive_count = 0
        
        for i in range(len(anomalies)):
            if anomalies[i]:
                consecutive_count += 1
            else:
                if consecutive_count < self.min_consecutive:
                    # Remove isolated anomalies
                    for j in range(max(0, i - consecutive_count), i):
                        filtered[j] = False
                consecutive_count = 0
        
        return filtered
    
    def _calculate_severity(self, scores: np.ndarray) -> np.ndarray:
        """Calculate anomaly severity levels"""
        
        severity = np.zeros(len(scores))
        
        for i, score in enumerate(scores):
            if score < self.threshold_sigma:
                severity[i] = 0  # Normal
            elif score < self.threshold_sigma * 1.5:
                severity[i] = 1  # Low
            elif score < self.threshold_sigma * 2:
                severity[i] = 2  # Medium
            else:
                severity[i] = 3  # High
        
        return severity
    
    def _generate_explanations(self, data: np.ndarray, 
                              scores: np.ndarray,
                              anomalies: np.ndarray) -> List[str]:
        """Generate human-readable explanations"""
        
        explanations = []
        
        for i in range(len(data)):
            if anomalies[i]:
                explanation = f"Anomaly at index {i}: "
                
                # Check type of anomaly
                if data[i] > self.historical_mean + 2 * self.historical_std:
                    explanation += "Unusually high consumption"
                elif data[i] < self.historical_mean - 2 * self.historical_std:
                    explanation += "Unusually low consumption"
                else:
                    explanation += "Irregular pattern detected"
                
                explanation += f" (score: {scores[i]:.2f})"
                explanations.append(explanation)
        
        return explanations


class AutoencoderAnomalyDetector(nn.Module):
    """Autoencoder-based anomaly detection"""
    
    def __init__(self, 
                 input_dim: int = 7,
                 encoding_dim: int = 3,
                 hidden_dims: List[int] = [32, 16]):
        
        super().__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = encoding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Reconstruction threshold (learned during training)
        self.threshold = None
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and encoding"""
        encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)
        return reconstruction, encoding
    
    def detect_anomalies(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies based on reconstruction error"""
        
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            
            # Calculate reconstruction error
            mse = F.mse_loss(reconstruction, x, reduction='none')
            reconstruction_error = torch.mean(mse, dim=-1)
            
            # Determine threshold if not set
            if self.threshold is None:
                self.threshold = torch.quantile(reconstruction_error, 0.95)
            
            # Detect anomalies
            anomalies = reconstruction_error > self.threshold
            scores = reconstruction_error / self.threshold
        
        return anomalies.cpu().numpy(), scores.cpu().numpy()


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detection methods"""
    
    def __init__(self, detectors: Optional[List] = None):
        
        if detectors is None:
            # Default ensemble
            self.detectors = [
                TSPulseDetector(),
                StatisticalDetector(),
                IsolationForestDetector()
            ]
        else:
            self.detectors = detectors
        
        self.weights = [1.0 / len(self.detectors)] * len(self.detectors)
        
    def fit(self, data: np.ndarray, **kwargs):
        """Fit all detectors"""
        for detector in self.detectors:
            if hasattr(detector, 'fit'):
                detector.fit(data, **kwargs)
        
    def detect(self, data: np.ndarray, **kwargs) -> AnomalyResult:
        """Ensemble anomaly detection"""
        
        all_scores = []
        all_anomalies = []
        
        for detector in self.detectors:
            if hasattr(detector, 'detect'):
                result = detector.detect(data, **kwargs)
                all_scores.append(result.scores)
                all_anomalies.append(result.anomalies)
            else:
                # Simple detection
                scores = detector.score_samples(data)
                anomalies = scores > detector.threshold
                all_scores.append(scores)
                all_anomalies.append(anomalies)
        
        # Weighted ensemble
        ensemble_scores = np.average(all_scores, axis=0, weights=self.weights)
        
        # Voting for anomalies
        ensemble_anomalies = np.sum(all_anomalies, axis=0) >= len(self.detectors) // 2
        
        # Severity based on agreement
        severity = np.sum(all_anomalies, axis=0) / len(self.detectors)
        
        # Generate explanations
        explanations = self._generate_ensemble_explanations(
            data, ensemble_scores, ensemble_anomalies, all_anomalies
        )
        
        # Statistics
        statistics = {
            'total_anomalies': np.sum(ensemble_anomalies),
            'anomaly_rate': np.mean(ensemble_anomalies) * 100,
            'detector_agreement': np.mean([np.array_equal(a, ensemble_anomalies) 
                                          for a in all_anomalies]) * 100
        }
        
        return AnomalyResult(
            scores=ensemble_scores,
            anomalies=ensemble_anomalies,
            severity=severity,
            explanations=explanations,
            statistics=statistics
        )
    
    def _generate_ensemble_explanations(self, data, scores, anomalies, 
                                       all_anomalies) -> List[str]:
        """Generate explanations from ensemble"""
        
        explanations = []
        detector_names = [type(d).__name__ for d in self.detectors]
        
        for i in range(len(data)):
            if anomalies[i]:
                detecting_methods = [name for j, name in enumerate(detector_names) 
                                   if all_anomalies[j][i]]
                
                explanation = (f"Anomaly detected by {len(detecting_methods)} methods: "
                             f"{', '.join(detecting_methods)} (score: {scores[i]:.2f})")
                
                explanations.append(explanation)
        
        return explanations


class StatisticalDetector:
    """Statistical anomaly detection methods"""
    
    def __init__(self, method='zscore', threshold=3.0):
        self.method = method
        self.threshold = threshold
        self.mean = None
        self.std = None
        
    def fit(self, data: np.ndarray, **kwargs):
        """Fit statistical parameters"""
        self.mean = np.mean(data)
        self.std = np.std(data)
        
    def detect(self, data: np.ndarray, **kwargs) -> AnomalyResult:
        """Detect statistical anomalies"""
        
        if self.method == 'zscore':
            scores = np.abs((data - self.mean) / (self.std + 1e-8))
        elif self.method == 'mad':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            scores = np.abs((data - median) / (mad + 1e-8))
        else:
            scores = np.zeros(len(data))
        
        anomalies = scores > self.threshold
        
        return AnomalyResult(
            scores=scores,
            anomalies=anomalies,
            severity=scores / self.threshold,
            explanations=[],
            statistics={'method': self.method}
        )


class IsolationForestDetector:
    """Isolation Forest based anomaly detection"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.threshold = None
        
    def fit(self, data: np.ndarray, **kwargs):
        """Fit isolation forest"""
        # Simplified version for demo
        self.threshold = np.percentile(data, (1 - self.contamination) * 100)
        
    def detect(self, data: np.ndarray, **kwargs) -> AnomalyResult:
        """Detect anomalies using isolation"""
        
        scores = np.abs(data - np.median(data))
        anomalies = scores > self.threshold if self.threshold else scores > np.percentile(scores, 90)
        
        return AnomalyResult(
            scores=scores,
            anomalies=anomalies,
            severity=scores / (self.threshold if self.threshold else 1),
            explanations=[],
            statistics={'contamination': self.contamination}
        )


class RealTimeAnomalyMonitor:
    """Real-time anomaly monitoring for production"""
    
    def __init__(self, detector, alert_callback=None):
        self.detector = detector
        self.alert_callback = alert_callback
        self.buffer = []
        self.anomaly_history = []
        
    def process_sample(self, value: float, timestamp: pd.Timestamp) -> Dict:
        """Process single sample in real-time"""
        
        self.buffer.append(value)
        
        # Keep buffer size manageable
        if len(self.buffer) > 1000:
            self.buffer.pop(0)
        
        # Detect anomaly
        result = self.detector.detect(np.array([value]))
        
        is_anomaly = result.anomalies[0]
        score = result.scores[0]
        
        # Track history
        self.anomaly_history.append({
            'timestamp': timestamp,
            'value': value,
            'is_anomaly': is_anomaly,
            'score': score
        })
        
        # Trigger alert if needed
        if is_anomaly and self.alert_callback:
            self.alert_callback({
                'timestamp': timestamp,
                'value': value,
                'score': score,
                'severity': result.severity[0]
            })
        
        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'severity': result.severity[0] if hasattr(result, 'severity') else 0
        }
    
    def get_recent_anomalies(self, hours: int = 24) -> pd.DataFrame:
        """Get recent anomalies"""
        
        if not self.anomaly_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.anomaly_history)
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        
        recent = df[df['timestamp'] > cutoff]
        anomalies = recent[recent['is_anomaly']]
        
        return anomalies


# Export classes
__all__ = [
    'TSPulseDetector',
    'AutoencoderAnomalyDetector',
    'EnsembleAnomalyDetector',
    'StatisticalDetector',
    'IsolationForestDetector',
    'RealTimeAnomalyMonitor',
    'AnomalyResult'
]