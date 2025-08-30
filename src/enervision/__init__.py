"""
EnerVision: AI-Powered Short-Term Energy Load Forecasting for Indian Buildings
Main implementation with IBM Granite TTM integration
"""

import os
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

# IBM watsonx.ai imports
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import TimeSeriesForecastingParams

# Time series imports
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results"""
    point_forecast: np.ndarray
    prediction_intervals: Dict[str, np.ndarray]
    anomaly_scores: np.ndarray
    optimization_opportunities: List[Dict]
    metadata: Dict[str, Any]


class TTMForecaster:
    """IBM Granite TTM Model Wrapper"""
    
    def __init__(self, model_size: str = "512-96"):
        self.model_size = model_size
        self.model = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """Load IBM Granite TTM model"""
        try:
            # For hackathon demo, using simulated model
            # In production, load from: 'ibm-granite/granite-timeseries-ttm-r2'
            logger.info(f"Loading IBM Granite TTM model: {self.model_size}")
            
            # Simulate model architecture
            self.config = {
                'context_length': int(self.model_size.split('-')[0]),
                'prediction_length': int(self.model_size.split('-')[1]),
                'channels': 7,  # Energy + 6 exogenous features
                'hidden_dim': 64,
                'num_layers': 4,
                'dropout': 0.1
            }
            
            # Initialize lightweight TTM architecture
            self.model = self._build_ttm_model()
            
        except Exception as e:
            logger.error(f"Error loading TTM model: {e}")
            raise
    
    def _build_ttm_model(self):
        """Build TinyTimeMixer architecture"""
        class TinyTimeMixer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Time mixing layers
                self.time_mixer = nn.Sequential(
                    nn.Linear(config['context_length'], config['hidden_dim']),
                    nn.ReLU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['hidden_dim'], config['hidden_dim'])
                )
                
                # Channel mixing layers
                self.channel_mixer = nn.Sequential(
                    nn.Linear(config['channels'], config['hidden_dim']),
                    nn.ReLU(),
                    nn.Dropout(config['dropout']),
                    nn.Linear(config['hidden_dim'], config['channels'])
                )
                
                # Prediction head
                self.prediction_head = nn.Linear(
                    config['hidden_dim'],
                    config['prediction_length']
                )
            
            def forward(self, x):
                # x shape: (batch, channels, context_length)
                batch_size = x.shape[0]
                
                # Time mixing
                time_mixed = self.time_mixer(x)
                
                # Channel mixing
                channel_mixed = self.channel_mixer(
                    time_mixed.transpose(1, 2)
                ).transpose(1, 2)
                
                # Generate predictions
                predictions = self.prediction_head(channel_mixed)
                
                return predictions
        
        return TinyTimeMixer(self.config)
    
    def forecast(self, historical_data: np.ndarray, 
                 exogenous_variables: Optional[np.ndarray] = None,
                 prediction_length: int = 96) -> np.ndarray:
        """Generate forecast using TTM model"""
        try:
            # Prepare input
            if exogenous_variables is not None:
                input_data = np.concatenate([
                    historical_data.reshape(-1, 1),
                    exogenous_variables
                ], axis=1)
            else:
                input_data = historical_data.reshape(-1, 1)
            
            # Ensure correct context length
            context_len = self.config['context_length']
            if len(input_data) > context_len:
                input_data = input_data[-context_len:]
            elif len(input_data) < context_len:
                # Pad with zeros if needed
                padding = np.zeros((context_len - len(input_data), input_data.shape[1]))
                input_data = np.vstack([padding, input_data])
            
            # Convert to tensor
            x = torch.FloatTensor(input_data).unsqueeze(0).transpose(1, 2)
            
            # Generate forecast
            with torch.no_grad():
                predictions = self.model(x)
            
            # Extract energy consumption predictions (first channel)
            forecast = predictions[0, 0, :prediction_length].numpy()
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error in TTM forecast: {e}")
            raise


class IBMEnsembleFusion:
    """Ensemble fusion layer for combining predictions"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'watsonx_api': 0.6,
            'local_ttm': 0.4
        }
    
    def combine_predictions(self, predictions: Dict[str, Any]) -> ForecastResult:
        """Combine multiple model predictions"""
        try:
            # Extract forecasts
            watsonx_forecast = predictions.get('watsonx_api', {}).get('forecast', np.array([]))
            local_forecast = predictions.get('local_ttm', np.array([]))
            
            # Ensure same length
            min_len = min(len(watsonx_forecast), len(local_forecast))
            if min_len == 0:
                # Fallback to single prediction
                if len(watsonx_forecast) > 0:
                    combined = watsonx_forecast
                else:
                    combined = local_forecast
            else:
                watsonx_forecast = watsonx_forecast[:min_len]
                local_forecast = local_forecast[:min_len]
                
                # Weighted ensemble
                combined = (
                    self.weights['watsonx_api'] * watsonx_forecast +
                    self.weights['local_ttm'] * local_forecast
                )
            
            # Calculate confidence intervals
            std_dev = np.std([watsonx_forecast, local_forecast], axis=0) if min_len > 0 else np.zeros_like(combined)
            confidence_intervals = {
                '95': (combined - 1.96 * std_dev, combined + 1.96 * std_dev),
                '80': (combined - 1.28 * std_dev, combined + 1.28 * std_dev)
            }
            
            # Anomaly scores (simplified)
            anomaly_scores = self._calculate_anomaly_scores(combined)
            
            # Optimization opportunities
            opportunities = self._identify_optimizations(combined)
            
            return ForecastResult(
                point_forecast=combined,
                prediction_intervals=confidence_intervals,
                anomaly_scores=anomaly_scores,
                optimization_opportunities=opportunities,
                metadata={
                    'ensemble_weights': self.weights,
                    'forecast_length': len(combined)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble fusion: {e}")
            raise
    
    def _calculate_anomaly_scores(self, forecast: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores using statistical methods"""
        # Simple z-score based anomaly detection
        mean = np.mean(forecast)
        std = np.std(forecast)
        z_scores = np.abs((forecast - mean) / (std + 1e-8))
        return z_scores
    
    def _identify_optimizations(self, forecast: np.ndarray) -> List[Dict]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Peak detection
        peaks = np.where(forecast > np.percentile(forecast, 90))[0]
        if len(peaks) > 0:
            opportunities.append({
                'type': 'peak_shaving',
                'time_indices': peaks.tolist(),
                'potential_savings': f"{len(peaks) * 0.15:.2f}%"
            })
        
        # Off-peak opportunities
        valleys = np.where(forecast < np.percentile(forecast, 30))[0]
        if len(valleys) > 0:
            opportunities.append({
                'type': 'load_shifting',
                'time_indices': valleys.tolist(),
                'potential_savings': f"{len(valleys) * 0.08:.2f}%"
            })
        
        return opportunities


class EnerVisionTTM:
    """Main EnerVision TTM Implementation"""
    
    def __init__(self, project_id: str = None, api_key: str = None):
        # Use provided credentials or environment variables
        self.project_id = project_id or os.environ.get("WATSONX_PROJECT_ID")
        self.api_key = api_key or os.environ.get("WATSONX_API_KEY")
        
        # Initialize watsonx.ai client
        self._init_watsonx_client()
        
        # Initialize TTM ensemble
        self.ttm_ensemble = {
            'ttm_512_96': TTMForecaster('512-96'),
            'ttm_1024_96': TTMForecaster('1024-96'),
            'ttm_1536_96': TTMForecaster('1536-96')
        }
        
        # Initialize fusion layer
        self.fusion_layer = IBMEnsembleFusion()
        
        logger.info("EnerVision TTM initialized successfully")
    
    def _init_watsonx_client(self):
        """Initialize IBM watsonx.ai client"""
        try:
            if self.api_key and self.project_id:
                credentials = Credentials(
                    url="https://us-south.ml.cloud.ibm.com",
                    api_key=self.api_key
                )
                self.watsonx_client = APIClient(credentials)
                self.watsonx_client.set.default_project(self.project_id)
                logger.info("watsonx.ai client initialized")
            else:
                logger.warning("watsonx.ai credentials not provided, using local models only")
                self.watsonx_client = None
        except Exception as e:
            logger.error(f"Error initializing watsonx.ai client: {e}")
            self.watsonx_client = None
    
    async def forecast_energy_load(self, 
                                   building_data: Dict,
                                   horizon: str = '24h') -> ForecastResult:
        """Generate ensemble forecasts using IBM Granite TTM models"""
        try:
            predictions = {}
            
            # Extract data
            consumption = np.array(building_data['consumption'])
            contextual_features = building_data.get('contextual_features')
            
            # Select appropriate TTM model based on context length
            context_length = len(consumption)
            if context_length <= 512:
                selected_model = self.ttm_ensemble['ttm_512_96']
            elif context_length <= 1024:
                selected_model = self.ttm_ensemble['ttm_1024_96']
            else:
                selected_model = self.ttm_ensemble['ttm_1536_96']
            
            prediction_length = self._convert_horizon_to_steps(horizon)
            
            # watsonx.ai API forecast (if available)
            if self.watsonx_client:
                try:
                    watsonx_forecast = await self._watsonx_api_forecast(
                        consumption, contextual_features, prediction_length
                    )
                    predictions['watsonx_api'] = {'forecast': watsonx_forecast}
                except Exception as e:
                    logger.warning(f"watsonx.ai API forecast failed: {e}")
            
            # Local TTM inference
            local_forecast = selected_model.forecast(
                historical_data=consumption,
                exogenous_variables=contextual_features,
                prediction_length=prediction_length
            )
            predictions['local_ttm'] = local_forecast
            
            # Ensemble combination
            ensemble_result = self.fusion_layer.combine_predictions(predictions)
            
            # Add anomaly detection
            ensemble_result.anomaly_scores = await self.detect_anomalies_with_tspulse(
                ensemble_result.point_forecast
            )
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error in energy load forecasting: {e}")
            raise
    
    async def _watsonx_api_forecast(self, 
                                    consumption: np.ndarray,
                                    features: Optional[np.ndarray],
                                    prediction_length: int) -> np.ndarray:
        """Generate forecast using watsonx.ai API"""
        # Simulate watsonx.ai API call for hackathon
        # In production, use actual API endpoint
        
        # Simple time series model simulation
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        try:
            # Ensure minimum data points
            if len(consumption) < 24:
                # Fallback to simple extrapolation
                return np.tile(consumption[-1], prediction_length)
            
            # Fit model
            model = ExponentialSmoothing(
                consumption,
                seasonal_periods=24,
                trend='add',
                seasonal='add'
            )
            fitted_model = model.fit(optimized=True)
            
            # Generate forecast
            forecast = fitted_model.forecast(prediction_length)
            
            return forecast
            
        except Exception as e:
            logger.warning(f"Fallback forecast due to error: {e}")
            # Simple fallback
            mean_consumption = np.mean(consumption)
            return np.full(prediction_length, mean_consumption)
    
    async def detect_anomalies_with_tspulse(self, forecast: np.ndarray) -> np.ndarray:
        """Detect anomalies using IBM Time Series Pulse approach"""
        # Simplified TSPulse-style anomaly detection
        
        # Calculate rolling statistics
        window_size = min(24, len(forecast) // 4)
        if window_size < 2:
            return np.zeros_like(forecast)
        
        # Rolling mean and std
        rolling_mean = pd.Series(forecast).rolling(window=window_size, center=True).mean()
        rolling_std = pd.Series(forecast).rolling(window=window_size, center=True).std()
        
        # Fill NaN values
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        
        # Calculate anomaly scores
        z_scores = np.abs((forecast - rolling_mean) / (rolling_std + 1e-8))
        
        # Normalize to 0-1 range
        anomaly_scores = z_scores / (z_scores.max() + 1e-8)
        
        return anomaly_scores
    
    def _convert_horizon_to_steps(self, horizon: str) -> int:
        """Convert horizon string to number of time steps"""
        horizon_map = {
            '15m': 1,
            '30m': 2,
            '1h': 4,
            '4h': 16,
            '12h': 48,
            '24h': 96,
            '48h': 192,
            '72h': 288
        }
        return horizon_map.get(horizon, 96)  # Default to 24h
    
    def identify_savings(self, forecast_result: ForecastResult) -> Dict:
        """Identify potential savings from forecast"""
        forecast = forecast_result.point_forecast
        
        # Calculate various savings metrics
        peak_reduction = self._calculate_peak_reduction_potential(forecast)
        load_shifting = self._calculate_load_shifting_savings(forecast)
        renewable_integration = self._calculate_renewable_optimization(forecast)
        
        total_savings = peak_reduction + load_shifting + renewable_integration
        
        return {
            'total_potential_savings': f"{total_savings:.1f}%",
            'peak_reduction': f"{peak_reduction:.1f}%",
            'load_shifting': f"{load_shifting:.1f}%",
            'renewable_integration': f"{renewable_integration:.1f}%",
            'annual_savings_inr': f"₹{total_savings * 25000:.0f}",  # Estimated per building
            'roi_months': max(3, int(24 - total_savings))
        }
    
    def _calculate_peak_reduction_potential(self, forecast: np.ndarray) -> float:
        """Calculate potential savings from peak reduction"""
        peak_threshold = np.percentile(forecast, 85)
        peak_hours = np.sum(forecast > peak_threshold)
        total_hours = len(forecast)
        return min(15.0, (peak_hours / total_hours) * 50)
    
    def _calculate_load_shifting_savings(self, forecast: np.ndarray) -> float:
        """Calculate potential savings from load shifting"""
        valley_threshold = np.percentile(forecast, 30)
        valley_hours = np.sum(forecast < valley_threshold)
        total_hours = len(forecast)
        return min(10.0, (valley_hours / total_hours) * 30)
    
    def _calculate_renewable_optimization(self, forecast: np.ndarray) -> float:
        """Calculate potential savings from renewable energy optimization"""
        # Simulate solar availability during day hours
        day_hours = len(forecast) // 3  # Assume 1/3 of time is sunny
        avg_day_consumption = np.mean(forecast[:day_hours])
        avg_total_consumption = np.mean(forecast)
        
        if avg_total_consumption > 0:
            renewable_potential = (avg_day_consumption / avg_total_consumption) * 20
            return min(10.0, renewable_potential)
        return 5.0  # Default potential


class IndianBuildingContextAdapter:
    """Adapter for Indian building-specific features"""
    
    def __init__(self):
        self.climate_zones = {
            'hot_dry': 0,
            'hot_humid': 1,
            'temperate': 2,
            'cold': 3
        }
        
        self.building_types = {
            'it_campus': 0,
            'commercial_office': 1,
            'mixed_use': 2,
            'educational': 3,
            'healthcare': 4,
            'retail': 5
        }
        
        # Indian festivals that impact energy consumption
        self.festivals = [
            'diwali', 'holi', 'dussehra', 'ganesh_chaturthi',
            'eid', 'christmas', 'independence_day', 'republic_day'
        ]
        
        self.granite_ttm = None
    
    async def fine_tune_for_indian_buildings(self, 
                                            building_dataset: Dict,
                                            model: Optional[TTMForecaster] = None) -> TTMForecaster:
        """Fine-tune IBM Granite TTM for Indian building patterns"""
        
        if model is None:
            model = TTMForecaster('512-96')
        
        # Create Indian-specific contextual features
        contextual_features = {
            'climate_zone': self._encode_indian_climate_zones(building_dataset.get('location', '')),
            'building_type': self._encode_indian_building_types(building_dataset.get('category', '')),
            'occupancy_pattern': self._extract_indian_occupancy_features(building_dataset),
            'seasonal_factors': self._extract_indian_seasonality(building_dataset),
            'festival_calendar': self._encode_indian_festivals(building_dataset.get('dates', [])),
            'monsoon_patterns': self._extract_monsoon_effects(building_dataset)
        }
        
        # Prepare training data
        consumption_data = np.array(building_dataset['consumption'])
        feature_matrix = self._create_feature_matrix(contextual_features, len(consumption_data))
        
        # Fine-tuning would happen here in production
        # For hackathon, we simulate the process
        logger.info(f"Fine-tuning TTM model for Indian context...")
        logger.info(f"Climate zone: {building_dataset.get('location', 'unknown')}")
        logger.info(f"Building type: {building_dataset.get('category', 'unknown')}")
        logger.info(f"Training data points: {len(consumption_data)}")
        
        # Update model with Indian context (simulated)
        model.config['indian_context'] = contextual_features
        
        return model
    
    def _encode_indian_climate_zones(self, location: str) -> int:
        """Encode Indian climate zones"""
        location = location.lower()
        
        if any(city in location for city in ['delhi', 'jaipur', 'ahmedabad']):
            return self.climate_zones['hot_dry']
        elif any(city in location for city in ['mumbai', 'chennai', 'kolkata']):
            return self.climate_zones['hot_humid']
        elif any(city in location for city in ['bangalore', 'pune', 'hyderabad']):
            return self.climate_zones['temperate']
        elif any(city in location for city in ['shimla', 'srinagar', 'darjeeling']):
            return self.climate_zones['cold']
        else:
            return self.climate_zones['temperate']  # Default
    
    def _encode_indian_building_types(self, category: str) -> int:
        """Encode Indian building types"""
        category = category.lower()
        
        for building_type, code in self.building_types.items():
            if building_type.replace('_', ' ') in category:
                return code
        
        return self.building_types['commercial_office']  # Default
    
    def _extract_indian_occupancy_features(self, dataset: Dict) -> np.ndarray:
        """Extract occupancy patterns specific to Indian buildings"""
        # Indian office patterns: 9-6 with lunch break at 1 PM
        # IT companies: 24/7 with shift patterns
        
        building_type = dataset.get('category', '').lower()
        
        if 'it' in building_type:
            # 24/7 operation with 3 shifts
            return np.array([0.8, 0.9, 0.7])  # Morning, afternoon, night occupancy
        elif 'educational' in building_type:
            # Academic schedule
            return np.array([0.9, 0.7, 0.1])  # High morning, moderate afternoon, low night
        else:
            # Standard office
            return np.array([0.9, 0.8, 0.2])  # Standard 9-6 pattern
    
    def _extract_indian_seasonality(self, dataset: Dict) -> np.ndarray:
        """Extract Indian seasonal patterns"""
        # Indian seasons: Summer (Mar-Jun), Monsoon (Jul-Sep), Post-monsoon (Oct-Nov), Winter (Dec-Feb)
        
        month = datetime.now().month
        
        if 3 <= month <= 6:  # Summer
            return np.array([1.3, 0.9, 1.0])  # High cooling, low heating, normal other
        elif 7 <= month <= 9:  # Monsoon
            return np.array([1.1, 0.7, 1.2])  # Moderate cooling, low heating, high dehumidification
        elif 10 <= month <= 11:  # Post-monsoon
            return np.array([1.0, 0.8, 0.9])  # Balanced
        else:  # Winter
            return np.array([0.7, 1.2, 0.8])  # Low cooling, high heating, normal other
    
    def _encode_indian_festivals(self, dates: List[str]) -> np.ndarray:
        """Encode Indian festival periods"""
        # Festivals significantly impact energy consumption
        festival_impact = np.zeros(len(self.festivals))
        
        for i, festival in enumerate(self.festivals):
            # Check if any date corresponds to festival period
            # Simplified for hackathon
            if festival in ['diwali', 'christmas']:
                festival_impact[i] = 1.5  # High impact
            elif festival in ['holi', 'eid']:
                festival_impact[i] = 1.2  # Moderate impact
            else:
                festival_impact[i] = 1.0  # Normal
        
        return festival_impact
    
    def _extract_monsoon_effects(self, dataset: Dict) -> np.ndarray:
        """Extract monsoon-specific patterns"""
        # Monsoon affects humidity, cooling requirements, and backup power usage
        
        location = dataset.get('location', '').lower()
        
        # Coastal cities have stronger monsoon effects
        if any(city in location for city in ['mumbai', 'chennai', 'kolkata']):
            return np.array([1.5, 1.3, 1.2])  # High humidity, increased HVAC, backup power
        elif any(city in location for city in ['delhi', 'jaipur']):
            return np.array([1.1, 1.0, 1.0])  # Moderate monsoon impact
        else:
            return np.array([1.2, 1.1, 1.1])  # Average impact
    
    def _create_feature_matrix(self, features: Dict, length: int) -> np.ndarray:
        """Create feature matrix for training"""
        feature_list = []
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if len(value) == length:
                    feature_list.append(value)
                else:
                    # Broadcast or tile to match length
                    repeated = np.tile(value, (length // len(value) + 1))[:length]
                    feature_list.append(repeated)
            else:
                # Single value, broadcast to array
                feature_list.append(np.full(length, value))
        
        return np.column_stack(feature_list)


# Export main classes
__all__ = [
    'EnerVisionTTM',
    'TTMForecaster',
    'IBMEnsembleFusion',
    'IndianBuildingContextAdapter',
    'ForecastResult'
]